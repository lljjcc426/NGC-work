from __future__ import annotations

import argparse
import json
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


DEFAULT_BASE = (
    ROOT
    / "reconstruction_candidates"
    / "b_task018_compact_source24_v6"
    / "task018.onnx"
)
DEFAULT_OUT = (
    ROOT
    / "reconstruction_candidates"
    / "b_task018_compact_target12_v8_safe"
    / "task018.onnx"
)


def _node(op_type: str, inputs: list[str], output: str, **attrs: object) -> onnx.NodeProto:
    return helper.make_node(op_type, inputs, [output], name=output, **attrs)


def rewrite(source: onnx.ModelProto) -> onnx.ModelProto:
    model = onnx.ModelProto.FromString(source.SerializeToString())
    model.graph.initializer.append(
        numpy_helper.from_array(np.asarray([12], dtype=np.int64), "task018_top12")
    )
    removed = {
        "point_anchors_150_0",
        "relative_points_151_0",
        "transformed_points_152_0",
        "target_for_points_153_0",
        "output_coords_154_0",
        "point_groups_row_155_0",
        "selected_group_157_0",
        "source_points_row_158_0",
        "selected_source_159_0",
        "target_valid_col_160_0",
        "selected_points_161_0",
        "selected_points_xy_166_0",
        "safe_coords_167_0",
        "source_colors_row_168_0",
        "safe_colors_169_0",
        "flat_coords_f16_170_0",
        "flat_coords_171_0",
        "flat_colors_172_0",
    }
    replacement = [
        _node("Unsqueeze", ["point_groups_argmin_149_0", "axes_zero"], "point_groups_row_155_0"),
        _node("Equal", ["point_groups_row_155_0", "task018_source_choice_col"], "selected_group_157_0"),
        _node("Unsqueeze", ["task018_source_valid", "axes_zero"], "source_points_row_158_0"),
        _node("And", ["selected_group_157_0", "source_points_row_158_0"], "selected_source_159_0"),
        _node(
            "Cast",
            ["selected_source_159_0"],
            "task018_target_source_scores",
            to=TensorProto.FLOAT16,
        ),
        helper.make_node(
            "TopK",
            ["task018_target_source_scores", "task018_top12"],
            ["task018_target_source_values", "task018_target_source_indices"],
            name="task018_target_source_top12",
            axis=1,
            largest=1,
            sorted=1,
        ),
        _node(
            "Gather",
            ["task018_source_coords", "task018_target_source_indices"],
            "task018_target_source_coords",
            axis=0,
        ),
        _node(
            "Gather",
            ["task018_source_colors", "task018_target_source_indices"],
            "task018_target_source_colors",
            axis=0,
        ),
        _node(
            "Gather",
            ["source_a_safe_68_0", "source_choice_141_0"],
            "task018_target_source_anchors",
            axis=0,
        ),
        _node(
            "Unsqueeze",
            ["task018_target_source_anchors", "axes_one"],
            "task018_target_source_anchors_col",
        ),
        _node(
            "Sub",
            ["task018_target_source_coords", "task018_target_source_anchors_col"],
            "relative_points_151_0",
        ),
        _node(
            "Einsum",
            ["relative_points_151_0", "selected_matrices_143_0"],
            "transformed_points_152_0",
            equation="tmi,tij->tmj",
        ),
        _node("Unsqueeze", ["target_a_coords_76_0", "axes_one"], "target_for_points_153_0"),
        _node(
            "Add",
            ["transformed_points_152_0", "target_for_points_153_0"],
            "task018_output_coords_raw",
        ),
        _node(
            "Clip",
            ["task018_output_coords_raw", "zero_f16", "max_index_f16"],
            "output_coords_154_0",
        ),
        _node(
            "Greater",
            ["task018_target_source_values", "zero_f16"],
            "task018_target_source_valid",
        ),
        _node("Unsqueeze", ["target_a_valid_77_0", "axes_one"], "target_valid_col_160_0"),
        _node(
            "And",
            ["task018_target_source_valid", "target_valid_col_160_0"],
            "selected_points_161_0",
        ),
        _node("Unsqueeze", ["selected_points_161_0", "axes_two"], "selected_points_xy_166_0"),
        _node(
            "Where",
            ["selected_points_xy_166_0", "output_coords_154_0", "dummy_coord_f16"],
            "safe_coords_167_0",
        ),
        _node(
            "Where",
            ["selected_points_161_0", "task018_target_source_colors", "zero_u8"],
            "safe_colors_169_0",
        ),
        _node("Reshape", ["safe_coords_167_0", "shape_coords"], "flat_coords_f16_170_0"),
        _node("Cast", ["flat_coords_f16_170_0"], "flat_coords_171_0", to=TensorProto.INT64),
        _node("Reshape", ["safe_colors_169_0", "shape_values"], "flat_colors_172_0"),
    ]

    nodes: list[onnx.NodeProto] = []
    seen: set[str] = set()
    inserted = False
    for source_node in model.graph.node:
        output = source_node.output[0] if source_node.output else ""
        if output == "point_anchors_150_0":
            nodes.extend(replacement)
            inserted = True
        if output in removed:
            seen.add(output)
            continue
        nodes.append(onnx.NodeProto.FromString(source_node.SerializeToString()))

    if not inserted or seen != removed:
        raise RuntimeError(
            f"task018 target point path mismatch: missing={sorted(removed - seen)} inserted={inserted}"
        )
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    oe.prune_dead(model)
    oe.prune_initializers(model)
    onnx.checker.check_model(model, full_check=True)
    return model


def compare(base: Path, candidate: Path) -> dict[str, int]:
    base_session = ort.InferenceSession(base.read_bytes(), providers=["CPUExecutionProvider"])
    candidate_session = ort.InferenceSession(candidate.read_bytes(), providers=["CPUExecutionProvider"])
    examples = json.loads((ROOT / "data" / "competition" / "task018.json").read_text())
    checked = 0
    for split in ("train", "test", "arc-gen"):
        for example in examples.get(split, []):
            pair = build_blend.convert_to_numpy(example)
            if pair is None:
                continue
            expected = base_session.run(["output"], {"input": pair["input"]})[0]
            actual = candidate_session.run(["output"], {"input": pair["input"]})[0]
            if not np.array_equal(expected, actual):
                raise RuntimeError(f"task018 compact equivalence failed in {split} example {checked}")
            checked += 1
    return {"checked": checked, "matched": checked}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(rewrite(onnx.load(args.base)), args.out)
    result = {
        "task": 18,
        "method": "12-point per-target source grouping before transformation",
        "equivalence": compare(args.base, args.out),
        "score": build_blend.validate_and_score((18, "task018_compact_target12", str(args.out))),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
