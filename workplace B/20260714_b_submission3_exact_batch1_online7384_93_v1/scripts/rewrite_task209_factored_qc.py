from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

import build_blend


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "team_baselines" / "team_submission2_20260713" / "submission" / "task209.onnx"
DEFAULT_OUT = ROOT / "reconstruction_candidates" / "b_task209_factored_qc_v1" / "task209.onnx"


def rewrite(base: onnx.ModelProto) -> onnx.ModelProto:
    model = onnx.ModelProto.FromString(base.SerializeToString())
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    qc = arrays["Qc"]
    scale_groups = np.zeros((3, 12, 5), dtype=np.float16)
    shifts = np.zeros((3, 5, 5), dtype=np.float16)
    for scale_index in range(3):
        scale = scale_index + 2
        for main_index in range(12):
            group = main_index // scale
            if group < 5:
                scale_groups[scale_index, main_index, group] = 1
    for shift in range(3):
        for group in range(5):
            pattern_index = shift + group
            if pattern_index < 5:
                shifts[shift, pattern_index, group] = 1
    reconstructed = np.einsum("smr,kpr->skmp", scale_groups, shifts)
    if not np.array_equal(qc, reconstructed):
        raise RuntimeError("Qc does not match the scale/shift factorization")

    kept = [item for item in model.graph.initializer if item.name != "Qc"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.initializer.extend([
        numpy_helper.from_array(scale_groups, "Qc_scale"),
        numpy_helper.from_array(shifts, "Qc_shift"),
    ])

    changed = 0
    for node in model.graph.node:
        if node.op_type != "Einsum" or "Qc" not in node.input:
            continue
        if list(node.input) not in (["BmRowHist", "Qc", "LmRow"], ["BmColHist", "Qc", "LmCol"]):
            raise RuntimeError("unexpected Qc Einsum inputs")
        left, _, right = node.input
        del node.input[:]
        node.input.extend([left, "Qc_scale", "Qc_shift", right])
        for attribute in node.attribute:
            if attribute.name == "equation":
                attribute.s = b"cm,smr,kpr,cp->sk"
        changed += 1
    if changed != 2:
        raise RuntimeError(f"expected two Qc contractions, found {changed}")

    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    return model


def compare_signs(base_path: Path, candidate_path: Path) -> dict[str, int]:
    base_session = ort.InferenceSession(base_path.read_bytes(), providers=["CPUExecutionProvider"])
    candidate_session = ort.InferenceSession(candidate_path.read_bytes(), providers=["CPUExecutionProvider"])
    examples = json.loads((ROOT / "data" / "competition" / "task209.json").read_text())
    checked = 0
    for split in ("train", "test", "arc-gen"):
        for example in examples.get(split, []):
            pair = build_blend.convert_to_numpy(example)
            if pair is None:
                continue
            expected = base_session.run(["output"], {"input": pair["input"]})[0] > 0
            actual = candidate_session.run(["output"], {"input": pair["input"]})[0] > 0
            if not np.array_equal(expected, actual):
                return {"right": checked, "total": checked + 1}
            checked += 1
    return {"right": checked, "total": checked}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(rewrite(onnx.load(args.base)), args.out)
    print(json.dumps({
        "score": build_blend.validate_and_score((209, "task209_factored_qc", str(args.out))),
        "official_sign_equivalence": compare_signs(args.base, args.out),
    }, indent=2))


if __name__ == "__main__":
    main()
