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


BASE = (
    ROOT
    / "public_probe_variants"
    / "team_submission4_b_work_20260715"
    / "submission"
    / "task063.onnx"
)
OUT = ROOT / "reconstruction_candidates" / "b_task063_u8_classes_v1" / "task063.onnx"


def build() -> onnx.ModelProto:
    model = onnx.load(BASE)
    replacements = {
        "z16": np.array([0], dtype=np.uint8),
        "o16": np.array([1], dtype=np.uint8),
        "t16": np.array([2], dtype=np.uint8),
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
        raise RuntimeError(f"missing class constants: {sorted(set(replacements) - found)}")

    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    model = build()
    onnx.save(model, OUT)
    result = build_blend.validate_and_score((63, "u8_classes", str(OUT)))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
