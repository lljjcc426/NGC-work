from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

import build_blend
import optimize_equivalent as oe


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = (
    ROOT
    / "public_probe_variants"
    / "team_pending_b018_b285_exact_v1_20260713"
    / "submission"
    / "task134.onnx"
)
DEFAULT_OUT = ROOT / "reconstruction_candidates" / "b_task134_y_variance_v1" / "task134.onnx"


def rewrite(base: onnx.ModelProto) -> onnx.ModelProto:
    model = onnx.ModelProto.FromString(base.SerializeToString())
    replaced = False
    for node in model.graph.node:
        if node.output and node.output[0] == "maxU":
            if len(node.input) != 2 or node.input[0] != "Uy":
                raise RuntimeError("task134 maxU node no longer has the expected inputs")
            node.input[1] = "Uy"
            replaced = True
            break
    if not replaced:
        raise RuntimeError("task134 maxU node was not found")

    oe.prune_dead(model)
    oe.prune_initializers(model)
    live_outputs = {name for node in model.graph.node for name in node.output}
    kept_value_info = [item for item in model.graph.value_info if item.name in live_outputs]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_value_info)
    onnx.checker.check_model(model, full_check=True)
    return model


def stress_test(path: Path, examples: int, seed: int) -> dict[str, int]:
    sys.path.insert(0, str(ROOT / "external" / "ARC-GEN"))
    from tasks import task_5ad4f10b  # type: ignore  # noqa: PLC0415

    session = ort.InferenceSession(path.read_bytes(), providers=["CPUExecutionProvider"])
    random.seed(seed)
    for index in range(examples):
        pair = build_blend.convert_to_numpy(task_5ad4f10b.generate())
        if pair is None:
            raise RuntimeError("generator produced an unsupported grid")
        output = session.run(["output"], {"input": pair["input"]})[0]
        if not np.array_equal((output > 0.0).astype(float), pair["output"]):
            return {"right": index, "total": index + 1, "first_failure": index}
    return {"right": examples, "total": examples, "first_failure": -1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stress", type=int, default=0)
    parser.add_argument("--seed", type=int, default=134_20260713)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(rewrite(onnx.load(args.base)), args.out)
    result: dict[str, object] = {
        "score": build_blend.validate_and_score((134, "task134_y_variance", str(args.out)))
    }
    if args.stress:
        result["stress"] = stress_test(args.out, args.stress, args.seed)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
