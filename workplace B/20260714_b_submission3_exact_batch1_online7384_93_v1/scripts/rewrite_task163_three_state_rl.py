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


DEFAULT_BASE = ROOT / "team_baselines" / "team_submission3_20260714" / "extracted" / "task163.onnx"
DEFAULT_SHARED = ROOT / "reconstruction_candidates" / "b_task163_shared_tables_v1" / "task163.onnx"
DEFAULT_OUT = ROOT / "reconstruction_candidates" / "b_task163_three_state_rl_v2" / "task163.onnx"


def rewrite(shared: onnx.ModelProto) -> onnx.ModelProto:
    model = onnx.ModelProto.FromString(shared.SerializeToString())
    if len(model.graph.node) != 1 or model.graph.node[0].op_type != "Einsum":
        raise RuntimeError("task163 shared model no longer has one terminal Einsum")

    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    rl = arrays.get("RL")
    selector = arrays.get("selector")
    if rl is None or rl.shape != (30, 4):
        raise RuntimeError("task163 RL no longer has the expected 30x4 shape")
    expected_selector = np.zeros((4, 3), dtype=np.float32)
    expected_selector[:3] = np.eye(3, dtype=np.float32)
    if selector is None or not np.array_equal(selector, expected_selector):
        raise RuntimeError("task163 selector no longer performs a three-state truncation")

    kept = [item for item in model.graph.initializer if item.name not in {"RL", "selector"}]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.initializer.append(numpy_helper.from_array(rl[:, :3].copy(), "RL"))

    node = helper.make_node(
        "Einsum",
        [
            "e4",
            "input",
            "RL", "TB",
            "RB", "TB", "RB", "RL", "RB", "RL",
            "e4",
            "input",
            "RL", "TB",
            "RB", "TB", "RB", "RL", "RB", "RL",
            "input", "V", "V", "A",
        ],
        ["output"],
        name="output",
        equation=(
            "a,nahq,hm,mxt,ho,oyt,rx,rd,iy,id,"
            "b,nbpg,gw,wut,gz,zvt,su,se,jv,je,"
            "ncij,kl,cf,lft->nkrs"
        ),
    )
    del model.graph.node[:]
    model.graph.node.append(node)
    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    return model


def official_equivalence(base: Path, candidate: Path) -> dict[str, float | int]:
    base_session = ort.InferenceSession(base.read_bytes(), providers=["CPUExecutionProvider"])
    candidate_session = ort.InferenceSession(candidate.read_bytes(), providers=["CPUExecutionProvider"])
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
                raise RuntimeError(f"official equivalence failed in {split} example {checked}")
            checked += 1
    return {"checked": checked, "matched": checked, "max_abs_error": max_abs_error}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--shared", type=Path, default=DEFAULT_SHARED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(rewrite(onnx.load(args.shared)), args.out)
    result = {
        "task": 163,
        "method": "remove explicit 4-to-3 selector and truncate RL globally",
        "equivalence": official_equivalence(args.base, args.out),
        "score": build_blend.validate_and_score((163, "task163_three_state_rl", str(args.out))),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
