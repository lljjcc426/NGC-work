from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import build_blend  # noqa: E402


DEFAULT_BASE = (
    ROOT
    / "reconstruction_candidates"
    / "b_task270_factored_selectors_v1"
    / "task270.onnx"
)
DEFAULT_OUT = (
    ROOT
    / "reconstruction_candidates"
    / "b_task270_factored_selectors_colors_v2"
    / "task270.onnx"
)


def rewrite(base: onnx.ModelProto) -> onnx.ModelProto:
    model = onnx.ModelProto.FromString(base.SerializeToString())
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    row_axis = arrays["selector_row_axis"]
    col_axis = arrays["selector_col_axis"]
    level = arrays["selector_level"]
    coef = arrays["coef"]

    center = np.zeros((11, 2), dtype=np.float16)
    center[:, 1] = 1
    center[(row_axis[:, 0] == 1) & (col_axis[:, 0] == 1), 0] = 1
    center[(row_axis[:, 0] == 1) & (col_axis[:, 0] == 1), 1] = 0

    color_rule = np.zeros((3, 2, 10), dtype=np.float16)
    assigned: dict[tuple[int, int], np.ndarray] = {}
    for index in range(11):
        level_index = int(np.argmax(level[index]))
        center_index = int(np.argmax(center[index]))
        key = (level_index, center_index)
        if key in assigned and not np.array_equal(assigned[key], coef[index]):
            raise RuntimeError(f"color rule is inconsistent for {key}")
        assigned[key] = coef[index]
        color_rule[key] = coef[index]
    reconstructed = np.einsum("kl,kz,lze->ke", level, center, color_rule)
    if not np.array_equal(coef, reconstructed):
        raise RuntimeError("color coefficient factorization failed")

    kept = [item for item in model.graph.initializer if item.name != "coef"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(center, "selector_center"),
            numpy_helper.from_array(color_rule, "color_rule"),
        ]
    )

    changed = 0
    for node in model.graph.node:
        if node.op_type != "Einsum" or list(node.output) != ["output"]:
            continue
        expected = [
            "Rconv",
            "E",
            "Cconv",
            "E",
            "selector_row_axis",
            "selector_level",
            "selector_col_axis",
            "selector_level",
            "flags",
            "coef",
        ]
        if list(node.input) != expected:
            raise RuntimeError(f"unexpected final Einsum inputs: {list(node.input)}")
        del node.input[:]
        node.input.extend(
            [
                "Rconv",
                "E",
                "Cconv",
                "E",
                "selector_row_axis",
                "selector_level",
                "selector_col_axis",
                "selector_level",
                "flags",
                "selector_level",
                "selector_center",
                "color_rule",
            ]
        )
        for attribute in node.attribute:
            if attribute.name == "equation":
                attribute.s = b"bcfr,rx,bghs,sy,kc,kf,kg,kh,bk,kl,kz,lze->bexy"
        changed += 1
    if changed != 1:
        raise RuntimeError(f"expected one final color Einsum, found {changed}")

    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    return model


def _equivalence(base: Path, candidate: Path) -> dict[str, int]:
    base_session = ort.InferenceSession(str(base), providers=["CPUExecutionProvider"])
    candidate_session = ort.InferenceSession(str(candidate), providers=["CPUExecutionProvider"])
    examples = json.loads((ROOT / "data" / "competition" / "task270.json").read_text())
    checked = 0
    for split in ("train", "test", "arc-gen"):
        for example in examples.get(split, []):
            pair = build_blend.convert_to_numpy(example)
            if pair is None:
                continue
            expected = base_session.run(["output"], {"input": pair["input"]})[0]
            actual = candidate_session.run(["output"], {"input": pair["input"]})[0]
            if not np.array_equal(expected, actual):
                raise RuntimeError(f"equivalence failed in {split} example {checked}")
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
        "task": 270,
        "method": "factor selectors and semantic color rules",
        "equivalence": _equivalence(args.base, args.out),
        "score": build_blend.validate_and_score((270, "task270_factored_colors", str(args.out))),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
