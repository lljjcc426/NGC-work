from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import build_blend  # noqa: E402
import optimize_equivalent as oe  # noqa: E402


DEFAULT_BASE = (
    ROOT
    / "team_baselines"
    / "team_submission2_20260713"
    / "submission"
    / "task328.onnx"
)
DEFAULT_OUT = (
    ROOT
    / "reconstruction_candidates"
    / "b_task328_dynamic_core_v1"
    / "task328.onnx"
)


def _add_initializer(model: onnx.ModelProto, name: str, value: np.ndarray) -> str:
    model.graph.initializer.append(numpy_helper.from_array(value, name=name))
    return name


def rewrite(model: onnx.ModelProto) -> onnx.ModelProto:
    inferred_base = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    base_types = {
        value.name: onnx.ValueInfoProto.FromString(value.SerializeToString())
        for value in list(inferred_base.graph.value_info) + list(inferred_base.graph.output)
    }
    current = onnx.ModelProto.FromString(model.SerializeToString())
    outputs = {out for node in current.graph.node for out in node.output}
    required = {"S_i64s", "coreLabel", "rIn", "cIn", "inside", "label", "labelPad"}
    if not required.issubset(outputs):
        raise RuntimeError(f"task328 graph mismatch: {sorted(required - outputs)}")

    shape1 = _add_initializer(current, "task328_shape1", np.array([1], dtype=np.int64))
    zero1 = _add_initializer(current, "task328_zero1", np.array([0], dtype=np.int64))
    axes_h = _add_initializer(current, "task328_axes_h", np.array([2], dtype=np.int64))
    axes_w = _add_initializer(current, "task328_axes_w", np.array([3], dtype=np.int64))
    thirty = _add_initializer(current, "task328_thirty", np.array(30, dtype=np.int64))
    pad_zeros = _add_initializer(current, "task328_pad_zeros", np.zeros(6, dtype=np.int64))

    setup = [
        helper.make_node("Reshape", ["S_i64s", shape1], ["task328_size1"], name="task328_size_vector"),
        helper.make_node(
            "Slice",
            ["aH", zero1, "task328_size1", axes_h],
            ["task328_aH"],
            name="task328_crop_rows",
        ),
        helper.make_node(
            "Slice",
            ["aW", zero1, "task328_size1", axes_w],
            ["task328_aW"],
            name="task328_crop_cols",
        ),
        helper.make_node(
            "Sub",
            [thirty, "S_i64s"],
            ["task328_pad_end_scalar"],
            name="task328_pad_end_scalar",
        ),
        helper.make_node(
            "Reshape",
            ["task328_pad_end_scalar", shape1],
            ["task328_pad_end"],
            name="task328_pad_end_vector",
        ),
        helper.make_node(
            "Concat",
            [pad_zeros, "task328_pad_end", "task328_pad_end"],
            ["task328_dynamic_pads"],
            name="task328_dynamic_pads",
            axis=0,
        ),
    ]

    remove_outputs = {"rIn", "cIn", "inside", "label"}
    rewritten: list[onnx.NodeProto] = []
    inserted = False
    for source_node in current.graph.node:
        node = onnx.NodeProto.FromString(source_node.SerializeToString())
        if any(output in remove_outputs for output in node.output):
            continue
        if "labelPad" in node.output:
            del node.input[:]
            node.input.extend(["coreLabel", "task328_dynamic_pads", "ten_u8"])
            node.name = "task328_pad_dynamic_core"
        else:
            for index, name in enumerate(node.input):
                if name == "aH":
                    node.input[index] = "task328_aH"
                elif name == "aW":
                    node.input[index] = "task328_aW"
        rewritten.append(node)
        if "S_i64s" in node.output:
            rewritten.extend(setup)
            inserted = True

    if not inserted:
        raise RuntimeError("failed to insert dynamic-size setup")

    del current.graph.node[:]
    current.graph.node.extend(rewritten)
    del current.graph.value_info[:]
    oe.prune_dead(current)
    oe.prune_initializers(current)

    # The contest scorer requires every intermediate to have a positive static
    # shape. Runtime Slice dimensions vary up to 18, so declare those maxima.
    output_names = {output for node in current.graph.node for output in node.output if output != "output"}
    for name in sorted(output_names):
        if name in base_types:
            current.graph.value_info.append(base_types[name])
    current.graph.value_info.extend(
        [
            helper.make_tensor_value_info("task328_size1", onnx.TensorProto.INT64, [1]),
            helper.make_tensor_value_info("task328_aH", onnx.TensorProto.UINT8, [1, 1, 18, 1]),
            helper.make_tensor_value_info("task328_aW", onnx.TensorProto.UINT8, [1, 1, 1, 18]),
            helper.make_tensor_value_info("task328_pad_end_scalar", onnx.TensorProto.INT64, []),
            helper.make_tensor_value_info("task328_pad_end", onnx.TensorProto.INT64, [1]),
            helper.make_tensor_value_info("task328_dynamic_pads", onnx.TensorProto.INT64, [8]),
        ]
    )
    onnx.checker.check_model(current, full_check=True)
    return current


def _official_equivalence(base: Path, candidate: Path) -> dict[str, int]:
    base_session = ort.InferenceSession(str(base), providers=["CPUExecutionProvider"])
    candidate_session = ort.InferenceSession(str(candidate), providers=["CPUExecutionProvider"])
    examples = json.loads((ROOT / "data" / "competition" / "task328.json").read_text())
    checked = 0
    for split in ("train", "test", "arc-gen"):
        for example in examples.get(split, []):
            pair = build_blend.convert_to_numpy(example)
            if pair is None:
                continue
            expected = base_session.run(["output"], {"input": pair["input"]})[0]
            actual = candidate_session.run(["output"], {"input": pair["input"]})[0]
            if not np.array_equal(expected > 0, actual > 0):
                raise RuntimeError(f"official equivalence failed in {split} example {checked}")
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
        "task": 328,
        "method": "dynamic-size Voronoi core with direct padding",
        "equivalence": _official_equivalence(args.base, args.out),
        "score": build_blend.validate_and_score((328, "task328_dynamic_core", str(args.out))),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
