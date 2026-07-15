from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import build_blend  # noqa: E402
import optimize_equivalent as oe  # noqa: E402


DEFAULT_BASE = (
    ROOT
    / "reconstruction_candidates"
    / "b_task076_even_sentinel_seed_v3"
    / "task076.onnx"
)
DEFAULT_OUT = (
    ROOT
    / "reconstruction_candidates"
    / "b_task076_visible_active_v4"
    / "task076.onnx"
)


def rewrite(source: onnx.ModelProto) -> onnx.ModelProto:
    model = onnx.ModelProto.FromString(source.SerializeToString())
    removed = {"source_visible_active", "cand_inactive", "cand_cell_ok"}
    nodes: list[onnx.NodeProto] = []
    seen: set[str] = set()
    for source_node in model.graph.node:
        output = source_node.output[0] if source_node.output else ""
        if output in removed:
            seen.add(output)
            continue
        node = onnx.NodeProto.FromString(source_node.SerializeToString())
        for index, name in enumerate(node.input):
            if name == "cand_cell_ok":
                node.input[index] = "cand_good"
        nodes.append(node)

    if seen != removed:
        raise RuntimeError(f"task076 visible active paths not found: {sorted(removed - seen)}")
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    oe.prune_dead(model)
    oe.prune_initializers(model)
    onnx.checker.check_model(model, full_check=True)
    return model


def compare(base: Path, candidate: Path) -> dict[str, int]:
    base_session = ort.InferenceSession(base.read_bytes(), providers=["CPUExecutionProvider"])
    candidate_session = ort.InferenceSession(candidate.read_bytes(), providers=["CPUExecutionProvider"])
    examples = json.loads((ROOT / "data" / "competition" / "task076.json").read_text())
    checked = 0
    for split in ("train", "test", "arc-gen"):
        for example in examples.get(split, []):
            pair = build_blend.convert_to_numpy(example)
            if pair is None:
                continue
            expected = base_session.run(["output"], {"input": pair["input"]})[0]
            actual = candidate_session.run(["output"], {"input": pair["input"]})[0]
            if not np.array_equal(expected, actual):
                raise RuntimeError(f"task076 baseline equivalence failed in {split} example {checked}")
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
        "task": 76,
        "method": "always-active six-cell source-visible TopK",
        "equivalence": compare(args.base, args.out),
        "score": build_blend.validate_and_score((76, "task076_visible_active", str(args.out))),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
