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

import build_blend
import optimize_equivalent as oe


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = (
    ROOT
    / "public_probe_variants"
    / "team_pending_b291_b368_b369_plus_b_tail5_20260713"
    / "submission"
    / "task123.onnx"
)
DEFAULT_OUT = (
    ROOT
    / "reconstruction_candidates"
    / "b_task123_square_rank_v1"
    / "task123.onnx"
)


def rewrite(base: onnx.ModelProto) -> onnx.ModelProto:
    model = onnx.ModelProto.FromString(base.SerializeToString())

    remove_initializers = {"canvas_index", "sentinel"}
    kept = [item for item in model.graph.initializer if item.name not in remove_initializers]
    if len(kept) != len(model.graph.initializer) - len(remove_initializers):
        raise RuntimeError("task123 canvas initializers were not found")
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)

    # U[a, r] is the indicator r <= a for the first ten rows. The difference
    # matrix telescopes the ten nested squares back into exactly one palette
    # color at radius max(row, col).
    u = np.zeros((10, 30), dtype=np.float16)
    for radius in range(10):
        u[radius, : radius + 1] = 1
    difference = np.zeros((10, 10), dtype=np.float16)
    for radius in range(9):
        difference[radius, radius] = 1
        difference[radius + 1, radius] = -1
    difference[9, 9] = 1
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(u, name="square_prefix"),
            numpy_helper.from_array(difference, name="square_difference"),
        ]
    )

    new_nodes: list[onnx.NodeProto] = []
    replaced_equal = False
    replaced_output = False
    for node in model.graph.node:
        if node.output and node.output[0] == "color_ids11":
            continue
        if node.output and node.output[0] == "color_sequence":
            updated = onnx.NodeProto.FromString(node.SerializeToString())
            updated.input[1] = "color_ids10"
            updated.output[0] = "color_sequence10"
            new_nodes.append(updated)
            new_nodes.append(
                helper.make_node(
                    "Cast",
                    ["color_sequence10"],
                    ["color_sequence_h"],
                    name="color_sequence_h",
                    to=TensorProto.FLOAT16,
                )
            )
            replaced_equal = True
            continue
        if node.output and node.output[0] == "output":
            new_nodes.append(
                helper.make_node(
                    "Einsum",
                    [
                        "color_sequence_h",
                        "square_difference",
                        "square_prefix",
                        "square_prefix",
                    ],
                    ["output"],
                    name="output",
                    equation="bcj,ja,ar,as->bcrs",
                )
            )
            replaced_output = True
            continue
        new_nodes.append(node)

    if not (replaced_equal and replaced_output):
        raise RuntimeError("task123 output path was not found")
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    del model.graph.value_info[:]
    oe.prune_dead(model)
    oe.prune_initializers(model)
    model.graph.output[0].type.tensor_type.elem_type = TensorProto.FLOAT16
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model


def stress_test(path: Path, examples: int, seed: int) -> tuple[int, int]:
    arc_gen_root = ROOT / "external" / "ARC-GEN"
    sys.path.insert(0, str(arc_gen_root))
    from tasks import task_539a4f51  # type: ignore  # noqa: PLC0415

    session = ort.InferenceSession(path.read_bytes(), providers=["CPUExecutionProvider"])
    random.seed(seed)
    for index in range(examples):
        pair = build_blend.convert_to_numpy(task_539a4f51.generate())
        if pair is None:
            raise RuntimeError("generator produced an unsupported grid")
        actual = session.run(["output"], {"input": pair["input"]})[0]
        if not np.array_equal(actual, pair["output"]):
            return index, index + 1
    return examples, examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stress", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123_20260713)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(rewrite(onnx.load(args.base)), args.out)
    result: dict[str, object] = {
        "score": build_blend.validate_and_score(
            (123, "task123_square_rank", str(args.out))
        )
    }
    if args.stress:
        right, total = stress_test(args.out, args.stress, args.seed)
        result["stress"] = {"right": right, "total": total, "seed": args.seed}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
