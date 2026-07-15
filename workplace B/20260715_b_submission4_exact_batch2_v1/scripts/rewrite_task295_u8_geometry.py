from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import build_blend  # noqa: E402


BASE = (
    ROOT
    / "public_probe_variants"
    / "team_submission4_b_work_20260715"
    / "submission"
    / "task295.onnx"
)
OUT = ROOT / "reconstruction_candidates" / "b_task295_u8_geometry_v1" / "task295.onnx"


def build() -> onnx.ModelProto:
    model = onnx.load(BASE)
    replacements = {
        "zero_f16": np.array(0, dtype=np.uint8),
        "row_index": np.arange(9, dtype=np.uint8).reshape(9, 1),
        "col_index": np.arange(18, dtype=np.uint8).reshape(1, 18),
        # Widths are guaranteed even by the task generator. Integer division is exact.
        "half_f16": np.array(2, dtype=np.uint8),
    }
    found = set()
    for index, initializer in enumerate(model.graph.initializer):
        value = replacements.get(initializer.name)
        if value is None:
            continue
        model.graph.initializer[index].CopyFrom(
            numpy_helper.from_array(value, initializer.name)
        )
        found.add(initializer.name)
    if found != set(replacements):
        raise RuntimeError(f"missing geometry constants: {sorted(set(replacements) - found)}")

    model.opset_import[0].version = 14
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(np.array([0, 2, 3], dtype=np.int64), "axes_counts"),
            numpy_helper.from_array(np.array([0], dtype=np.int64), "axes_zero"),
        ]
    )

    changed_casts = set()
    changed_half = False
    for node in model.graph.node:
        if node.op_type == "ReduceSum":
            axes_name = "axes_counts" if list(node.output) == ["counts"] else "axes_zero"
            node.input.append(axes_name)
            kept = [attr for attr in node.attribute if attr.name != "axes"]
            del node.attribute[:]
            node.attribute.extend(kept)
        if list(node.output) in (["width_f16"], ["length_f16"]):
            attr = next(attr for attr in node.attribute if attr.name == "to")
            attr.i = TensorProto.UINT8
            changed_casts.add(node.output[0])
        if list(node.output) == ["half_width"]:
            if node.op_type != "Mul":
                raise RuntimeError(f"unexpected half-width op: {node.op_type}")
            node.op_type = "Div"
            changed_half = True
    if changed_casts != {"width_f16", "length_f16"} or not changed_half:
        raise RuntimeError("integer geometry rewrite was incomplete")

    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    model = build()
    onnx.save(model, OUT)
    result = build_blend.validate_and_score((295, "u8_geometry", str(OUT)))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
