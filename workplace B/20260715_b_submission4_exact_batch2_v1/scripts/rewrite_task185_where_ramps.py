from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import build_blend  # noqa: E402
import optimize_equivalent as oe  # noqa: E402


BASE = (
    ROOT
    / "public_probe_variants"
    / "team_submission4_b_work_20260715"
    / "submission"
    / "task185.onnx"
)
OUT = ROOT / "reconstruction_candidates" / "b_task185_where_ramps_v1" / "task185.onnx"


def build() -> onnx.ModelProto:
    model = onnx.load(BASE)
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name not in {"ramp_desc", "ramp_asc"}:
            continue
        value = numpy_helper.to_array(initializer).astype(np.uint8)
        model.graph.initializer[index].CopyFrom(
            numpy_helper.from_array(value, initializer.name)
        )
    model.graph.initializer.append(
        numpy_helper.from_array(np.array(0, dtype=np.uint8), "zero_i8")
    )

    replacements = {
        "rdesc": ("rbin_b", "ramp_desc", "zero_i8"),
        "rasc": ("rbin_b", "ramp_asc", "zero_i8"),
        "cdesc": ("cbin_b", "ramp_desc", "zero_i8"),
    }
    found = set()
    for node in model.graph.node:
        if len(node.output) != 1 or node.output[0] not in replacements:
            continue
        if node.op_type != "Mul":
            raise RuntimeError(f"unexpected ramp op for {node.output[0]}: {node.op_type}")
        node.op_type = "Where"
        del node.input[:]
        node.input.extend(replacements[node.output[0]])
        found.add(node.output[0])
    if found != set(replacements):
        raise RuntimeError(f"missing ramp nodes: {sorted(set(replacements) - found)}")

    oe.prune_dead(model)
    oe.prune_initializers(model)
    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    model = build()
    onnx.save(model, OUT)
    result = build_blend.validate_and_score((185, "where_ramps", str(OUT)))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
