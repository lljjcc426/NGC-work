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
DEFAULT_BASE = ROOT / "team_baselines" / "team_submission2_20260713" / "submission" / "task360.onnx"
DEFAULT_OUT = ROOT / "reconstruction_candidates" / "b_task360_shared_route_v1" / "task360.onnx"


def rewrite(base: onnx.ModelProto) -> onnx.ModelProto:
    model = onnx.ModelProto.FromString(base.SerializeToString())
    if len(model.graph.node) != 1 or model.graph.node[0].op_type != "Einsum":
        raise RuntimeError("task360 baseline no longer has the expected single Einsum")

    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    left_fold = arrays["left_fold"]
    route = arrays["route_to_output"]
    output_gate = np.zeros(30, dtype=np.float32)
    output_gate[:4] = 1
    reconstructed = left_fold.T * output_gate.reshape(1, 30)
    if not np.array_equal(route, reconstructed):
        raise RuntimeError("task360 route sharing identity no longer holds")

    kept = [item for item in model.graph.initializer if item.name != "route_to_output"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.initializer.append(numpy_helper.from_array(output_gate, "output_gate"))

    node = helper.make_node(
        "Einsum",
        ["input", "channel", "left_fold", "left_fold", "output_gate"],
        ["output"],
        name="output",
        equation="bchw,oc,wk,vk,v->bohv",
    )
    del model.graph.node[:]
    model.graph.node.append(node)
    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    return model


def compare_signs(base_path: Path, candidate_path: Path) -> dict[str, int]:
    base_session = ort.InferenceSession(base_path.read_bytes(), providers=["CPUExecutionProvider"])
    candidate_session = ort.InferenceSession(candidate_path.read_bytes(), providers=["CPUExecutionProvider"])
    examples = json.loads((ROOT / "data" / "competition" / "task360.json").read_text())
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
    result = {
        "score": build_blend.validate_and_score((360, "task360_shared_route", str(args.out))),
        "official_sign_equivalence": compare_signs(args.base, args.out),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
