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
DEFAULT_BASE = ROOT / "team_baselines" / "team_submission2_20260713" / "submission" / "task163.onnx"
DEFAULT_OUT = ROOT / "reconstruction_candidates" / "b_task163_shared_tables_v1" / "task163.onnx"


def rewrite(base: onnx.ModelProto) -> onnx.ModelProto:
    model = onnx.ModelProto.FromString(base.SerializeToString())
    if len(model.graph.node) != 1 or model.graph.node[0].op_type != "Einsum":
        raise RuntimeError("task163 baseline no longer has the expected single Einsum")

    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    e4 = arrays["e4"]
    tl = arrays["TL"]
    tb = arrays["TB"]
    selector = np.zeros((4, 3), dtype=np.float32)
    selector[:3, :] = np.eye(3, dtype=np.float32)

    if float(e4.sum()) != 1.0:
        raise RuntimeError("the color selector e4 is malformed")
    if not np.array_equal(tl, np.einsum("mp,pxt->mxt", selector, tb)):
        raise RuntimeError("TL cannot be reconstructed exactly from selector and TB")

    kept = [item for item in model.graph.initializer if item.name != "TL"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.initializer.append(numpy_helper.from_array(selector, "selector"))

    node = helper.make_node(
        "Einsum",
        [
            "e4",
            "input",
            "RL", "selector", "TB",
            "RB", "TB", "RB", "RL", "RB", "RL",
            "e4",
            "input",
            "RL", "selector", "TB",
            "RB", "TB", "RB", "RL", "RB", "RL",
            "input", "V", "V", "A",
        ],
        ["output"],
        name="output",
        equation=(
            "a,nahq,hm,mA,Axt,ho,oyt,rx,rd,iy,id,"
            "b,nbpg,gw,wB,But,gz,zvt,su,se,jv,je,"
            "ncij,kl,cf,lft->nkrs"
        ),
    )
    del model.graph.node[:]
    model.graph.node.append(node)
    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    return model


def compare_outputs(base_path: Path, candidate_path: Path) -> dict[str, float | int]:
    base_session = ort.InferenceSession(base_path.read_bytes(), providers=["CPUExecutionProvider"])
    candidate_session = ort.InferenceSession(candidate_path.read_bytes(), providers=["CPUExecutionProvider"])
    examples = json.loads((ROOT / "data" / "competition" / "task163.json").read_text())
    checked = 0
    max_abs_error = 0.0
    for split in ("train", "test", "arc-gen"):
        for example in examples.get(split, []):
            pair = build_blend.convert_to_numpy(example)
            if pair is None:
                continue
            expected = base_session.run(["output"], {"input": pair["input"]})[0]
            actual = candidate_session.run(["output"], {"input": pair["input"]})[0]
            max_abs_error = max(max_abs_error, float(np.max(np.abs(expected - actual))))
            if not np.array_equal(expected > 0, actual > 0):
                return {"right": checked, "total": checked + 1, "max_abs_error": max_abs_error}
            checked += 1
    return {"right": checked, "total": checked, "max_abs_error": max_abs_error}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(rewrite(onnx.load(args.base)), args.out)
    result = {
        "score": build_blend.validate_and_score((163, "task163_shared_tables", str(args.out))),
        "official_sign_equivalence": compare_outputs(args.base, args.out),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
