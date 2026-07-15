from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import build_blend  # noqa: E402
import optimize_equivalent as oe  # noqa: E402
from rewrite_task018_exact_four_transforms import official_check  # noqa: E402


DEFAULT_BASE = (
    ROOT
    / "public_probe_variants"
    / "team_submission5_b_work_20260716"
    / "submission"
    / "task018.onnx"
)


def _node(op_type: str, inputs: list[str], output: str, **attrs: object) -> onnx.NodeProto:
    return helper.make_node(op_type, inputs, [output], name=output, **attrs)


def rewrite(source: onnx.ModelProto) -> onnx.ModelProto:
    model = onnx.ModelProto.FromString(source.SerializeToString())
    model.graph.initializer.append(
        numpy_helper.from_array(
            np.array([24, 1], dtype=np.float16),
            "task018_flat_weights",
        )
    )
    nodes = list(model.graph.node)

    for marker, candidate, first, clipped, in_bounds in (
        (
            "b",
            "candidate_b_99_0",
            "match_b_ge_zero_101_0",
            "match_b_clipped_107_0",
            "match_b_in_bounds_b_106_0",
        ),
        (
            "c",
            "candidate_c_100_0",
            "match_c_ge_zero_116_0",
            "match_c_clipped_122_0",
            "match_c_in_bounds_b_121_0",
        ),
    ):
        start = next(index for index, node in enumerate(nodes) if first in node.output)
        end = next(index for index, node in enumerate(nodes) if clipped in node.output) + 1
        replacement = [
            _node("Clip", [candidate, "zero_f16", "max_index_f16"], clipped),
            _node("Equal", [candidate, clipped], f"task018_{marker}_clip_equal_xy"),
            _node(
                "Gather",
                [f"task018_{marker}_clip_equal_xy", "coord0"],
                f"task018_{marker}_row_in_bounds",
                axis=3,
            ),
            _node(
                "Gather",
                [f"task018_{marker}_clip_equal_xy", "coord1"],
                f"task018_{marker}_col_in_bounds",
                axis=3,
            ),
            _node(
                "And",
                [f"task018_{marker}_row_in_bounds", f"task018_{marker}_col_in_bounds"],
                in_bounds,
            ),
        ]
        nodes[start:end] = replacement

    for marker, rows, flat in (
        ("b", "match_b_rows_108_0", "match_b_flat_f16_111_0"),
        ("c", "match_c_rows_123_0", "match_c_flat_f16_126_0"),
    ):
        start = next(index for index, node in enumerate(nodes) if rows in node.output)
        end = next(index for index, node in enumerate(nodes) if flat in node.output) + 1
        clipped = f"match_{marker}_clipped_{107 if marker == 'b' else 122}_0"
        nodes[start:end] = [
            helper.make_node(
                "Einsum",
                [clipped, "task018_flat_weights"],
                [flat],
                name=flat,
                equation="...i,i->...",
            )
        ]

    group_start = next(
        index for index, node in enumerate(nodes) if "point_groups_left_144_0" in node.output
    )
    group_end = next(
        index for index, node in enumerate(nodes) if "selected_group_157_0" in node.output
    ) + 1
    nodes[group_start:group_end] = [
        _node("Gather", ["source_a_safe_68_0", "coord0"], "task018_group_anchor0", axis=0),
        _node("Gather", ["source_a_safe_68_0", "coord1"], "task018_group_anchor1", axis=0),
        _node(
            "Sub",
            ["task018_source_coords", "task018_group_anchor0"],
            "task018_group_delta0",
        ),
        _node("Abs", ["task018_group_delta0"], "task018_group_abs0"),
        _node(
            "ReduceSum",
            ["task018_group_abs0", "axes_one"],
            "task018_group_distance0",
            keepdims=0,
        ),
        _node(
            "Sub",
            ["task018_source_coords", "task018_group_anchor1"],
            "task018_group_delta1",
        ),
        _node("Abs", ["task018_group_delta1"], "task018_group_abs1"),
        _node(
            "ReduceSum",
            ["task018_group_abs1", "axes_one"],
            "task018_group_distance1",
            keepdims=0,
        ),
        _node(
            "Less",
            ["task018_group_distance1", "task018_group_distance0"],
            "task018_group_is_one",
        ),
        _node(
            "Cast",
            ["source_choice_141_0"],
            "task018_source_choice_bool",
            to=TensorProto.BOOL,
        ),
        _node(
            "Unsqueeze",
            ["task018_source_choice_bool", "axes_one"],
            "task018_source_choice_bool_col",
        ),
        _node(
            "Equal",
            ["task018_group_is_one", "task018_source_choice_bool_col"],
            "selected_group_157_0",
        ),
    ]

    flat_coords_reshape = next(
        node for node in nodes if "flat_coords_f16_170_0" in node.output
    )
    flat_coords_cast = next(node for node in nodes if "flat_coords_171_0" in node.output)
    flat_colors_reshape = next(node for node in nodes if "flat_colors_172_0" in node.output)
    scatter = next(node for node in nodes if "stamped_177_0" in node.output)
    if (
        flat_coords_reshape.op_type != "Reshape"
        or flat_coords_cast.op_type != "Cast"
        or flat_colors_reshape.op_type != "Reshape"
        or scatter.op_type != "ScatterND"
    ):
        raise RuntimeError("task018 flattened ScatterND path was not found")
    flat_coords_cast.input[0] = "safe_coords_167_0"
    scatter.input[1] = "flat_coords_171_0"
    scatter.input[2] = "safe_colors_169_0"
    nodes = [
        node
        for node in nodes
        if node is not flat_coords_reshape and node is not flat_colors_reshape
    ]

    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.value_info[:]
    oe.prune_dead(model)
    oe.prune_initializers(model)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model


def stress_equivalence(
    base: Path,
    candidate: Path,
    examples: int,
    seed: int,
    reseed_every: int,
) -> dict[str, int]:
    sys.path.insert(0, str(ROOT / "third_party" / "ARC-GEN"))
    from tasks import task_0e206a2e  # type: ignore  # noqa: PLC0415

    base_session = ort.InferenceSession(base.read_bytes(), providers=["CPUExecutionProvider"])
    candidate_session = ort.InferenceSession(
        candidate.read_bytes(), providers=["CPUExecutionProvider"]
    )
    for index in range(examples):
        if index % reseed_every == 0:
            random.seed(seed + index // reseed_every)
        example = task_0e206a2e.generate()
        pair = build_blend.convert_to_numpy(example)
        if pair is None:
            raise RuntimeError("generator produced an unsupported grid")
        expected = base_session.run(["output"], {"input": pair["input"]})[0]
        actual = candidate_session.run(["output"], {"input": pair["input"]})[0]
        if not np.array_equal(expected, actual):
            raise RuntimeError(f"fresh baseline equivalence failed at example {index}")
        if (index + 1) % 1000 == 0:
            print(f"equivalence_stress={index + 1}/{examples}", flush=True)
    return {
        "checked": examples,
        "matched": examples,
        "seed": seed,
        "reseed_every": reseed_every,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stress", type=int, default=0)
    parser.add_argument("--seed", type=int, default=18_20260716)
    parser.add_argument("--reseed-every", type=int, default=50)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(rewrite(onnx.load(args.base)), args.out)
    result = {
        "task": 18,
        "method": (
            "clip bounds, linearized lookup, direct two-anchor grouping, "
            "and rank-3 ScatterND"
        ),
        "official": official_check(args.out),
        "stress": (
            stress_equivalence(args.base, args.out, args.stress, args.seed, args.reseed_every)
            if args.stress
            else None
        ),
        "score": build_blend.validate_and_score((18, "task018_clip_bounds", str(args.out))),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
