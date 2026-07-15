from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import build_blend  # noqa: E402
import optimize_equivalent as oe  # noqa: E402


DEFAULT_BASE = (
    ROOT
    / "reconstruction_candidates"
    / "b_task101_signed_color_fusion_v1"
    / "task101.onnx"
)
DEFAULT_OUT = (
    ROOT
    / "reconstruction_candidates"
    / "b_task101_binary_topk_v2"
    / "task101.onnx"
)


def rewrite(source: onnx.ModelProto) -> onnx.ModelProto:
    model = onnx.ModelProto.FromString(source.SerializeToString())
    mask_by_output = {
        "aX": "aT",
        "a4": "aW",
        "bc": "3",
    }
    validity_outputs = {"a0", "bf"}
    changed: set[str] = set()
    nodes: list[onnx.NodeProto] = []

    for source_node in model.graph.node:
        node = onnx.NodeProto.FromString(source_node.SerializeToString())
        output = node.output[0] if node.output else ""
        if output in mask_by_output:
            if node.op_type != "Where" or node.input[0] != mask_by_output[output]:
                raise RuntimeError(f"unexpected task101 rank scorer: {output}")
            nodes.append(
                helper.make_node(
                    "Cast",
                    [mask_by_output[output]],
                    [output],
                    name=output,
                    to=TensorProto.FLOAT16,
                )
            )
            changed.add(output)
            continue
        if output in validity_outputs:
            if node.op_type != "Greater" or node.input[1] != "E":
                raise RuntimeError(f"unexpected task101 TopK validity check: {output}")
            node.input[1] = "i"
            changed.add(output)
        nodes.append(node)

    expected = set(mask_by_output) | validity_outputs
    if changed != expected:
        raise RuntimeError(f"task101 ranking paths not found: {sorted(expected - changed)}")
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    oe.prune_dead(model)
    oe.prune_initializers(model)
    onnx.checker.check_model(model, full_check=True)
    return model


def compare(base: Path, candidate: Path) -> dict[str, int]:
    base_session = ort.InferenceSession(base.read_bytes(), providers=["CPUExecutionProvider"])
    candidate_session = ort.InferenceSession(candidate.read_bytes(), providers=["CPUExecutionProvider"])
    examples = json.loads((ROOT / "data" / "competition" / "task101.json").read_text())
    checked = 0
    for split in ("train", "test", "arc-gen"):
        for example in examples.get(split, []):
            pair = build_blend.convert_to_numpy(example)
            if pair is None:
                continue
            expected = base_session.run(["output"], {"input": pair["input"]})[0]
            actual = candidate_session.run(["output"], {"input": pair["input"]})[0]
            if not np.array_equal(expected, actual):
                raise RuntimeError(f"task101 baseline equivalence failed in {split} example {checked}")
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
        "task": 101,
        "method": "direct float16 binary TopK selection",
        "equivalence": compare(args.base, args.out),
        "score": build_blend.validate_and_score((101, "task101_binary_topk", str(args.out))),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
