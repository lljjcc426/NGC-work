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
    / "team_baselines"
    / "team_submission2_20260713"
    / "submission"
    / "task270.onnx"
)
DEFAULT_OUT = (
    ROOT
    / "reconstruction_candidates"
    / "b_task270_factored_selectors_v1"
    / "task270.onnx"
)


def _factor_selector(selector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if selector.shape != (11, 3, 3):
        raise RuntimeError(f"unexpected selector shape {selector.shape}")
    left = np.zeros((11, 3), dtype=np.float16)
    right = np.zeros((11, 3), dtype=np.float16)
    for index, plane in enumerate(selector):
        nonzero = np.argwhere(plane != 0)
        if nonzero.shape != (1, 2) or plane[tuple(nonzero[0])] != 1:
            raise RuntimeError("selector plane is not one-hot")
        row, col = nonzero[0]
        left[index, row] = 1
        right[index, col] = 1
    if not np.array_equal(selector, np.einsum("kc,kf->kcf", left, right)):
        raise RuntimeError("selector factorization failed")
    return left, right


def rewrite(base: onnx.ModelProto) -> onnx.ModelProto:
    model = onnx.ModelProto.FromString(base.SerializeToString())
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    selr_c, selr_f = _factor_selector(arrays["selr"])
    selc_g, selc_h = _factor_selector(arrays["selc"])
    if not np.array_equal(selr_f, selc_h):
        raise RuntimeError("row and column level selectors are no longer shared")

    kept = [item for item in model.graph.initializer if item.name not in {"selr", "selc"}]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(selr_c, "selector_row_axis"),
            numpy_helper.from_array(selc_g, "selector_col_axis"),
            numpy_helper.from_array(selr_f, "selector_level"),
        ]
    )

    changed = 0
    for node in model.graph.node:
        if node.op_type != "Einsum" or list(node.output) != ["output"]:
            continue
        expected = ["Rconv", "E", "Cconv", "E", "selr", "selc", "flags", "coef"]
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
                "coef",
            ]
        )
        for attribute in node.attribute:
            if attribute.name == "equation":
                attribute.s = b"bcfr,rx,bghs,sy,kc,kf,kg,kh,bk,ke->bexy"
        changed += 1
    if changed != 1:
        raise RuntimeError(f"expected one final selector Einsum, found {changed}")

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
        "method": "factor and share one-hot selector axes",
        "equivalence": _equivalence(args.base, args.out),
        "score": build_blend.validate_and_score((270, "task270_factored_selectors", str(args.out))),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
