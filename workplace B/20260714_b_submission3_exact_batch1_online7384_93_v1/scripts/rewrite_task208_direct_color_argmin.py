from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import build_blend  # noqa: E402
import optimize_equivalent as oe  # noqa: E402


DEFAULT_BASE = ROOT / "team_baselines" / "team_submission3_20260714" / "extracted" / "task208.onnx"
DEFAULT_COMPACT = ROOT / "reconstruction_candidates" / "b_task208_compact_search_v1" / "task208.onnx"
DEFAULT_OUT = ROOT / "reconstruction_candidates" / "b_task208_direct_color_argmin_v2" / "task208.onnx"


def rewrite(compact: onnx.ModelProto) -> onnx.ModelProto:
    model = onnx.ModelProto.FromString(compact.SerializeToString())
    outputs = {output for node in model.graph.node for output in node.output}
    required = {"c19", "present", "score", "boxidx", "boxc"}
    if not required.issubset(outputs):
        raise RuntimeError(f"task208 compact graph mismatch: {sorted(required - outputs)}")

    rewritten: list[onnx.NodeProto] = []
    for source in model.graph.node:
        node = onnx.NodeProto.FromString(source.SerializeToString())
        if "c19" in node.output or "boxc" in node.output:
            continue
        for index, name in enumerate(node.input):
            if name == "c19":
                node.input[index] = "cnt"
        if "boxidx" in node.output:
            node.output[0] = "boxc"
            node.name = "task208_direct_color_argmin"
        rewritten.append(node)

    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    del model.graph.value_info[:]
    oe.prune_dead(model)
    oe.prune_initializers(model)
    onnx.checker.check_model(model, full_check=True)
    return model


def official_equivalence(base: Path, candidate: Path) -> tuple[dict[str, int], int]:
    base_session = ort.InferenceSession(base.read_bytes(), providers=["CPUExecutionProvider"])
    candidate_session = ort.InferenceSession(candidate.read_bytes(), providers=["CPUExecutionProvider"])
    examples = json.loads((ROOT / "data" / "competition" / "task208.json").read_text())
    checked = 0
    minimum_margin = 10**9
    for split in ("train", "test", "arc-gen"):
        for example in examples.get(split, []):
            pair = build_blend.convert_to_numpy(example)
            if pair is None:
                continue
            expected = base_session.run(["output"], {"input": pair["input"]})[0]
            actual = candidate_session.run(["output"], {"input": pair["input"]})[0]
            if not np.array_equal(expected > 0, actual > 0):
                raise RuntimeError(f"official equivalence failed in {split} example {checked}")

            counts = Counter(value for row in example["input"] for value in row)
            nonzero = [(count, color) for color, count in counts.items() if color]
            box_count, _ = min(nonzero)
            minimum_margin = min(minimum_margin, counts[0] - box_count)
            checked += 1
    return {"checked": checked, "matched": checked}, minimum_margin


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--compact", type=Path, default=DEFAULT_COMPACT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(rewrite(onnx.load(args.compact)), args.out)
    equivalence, minimum_margin = official_equivalence(args.base, args.out)
    result = {
        "task": 208,
        "method": "direct ten-channel ArgMin under exact generator count separation",
        "equivalence": equivalence,
        "official_min_background_margin": minimum_margin,
        "score": build_blend.validate_and_score((208, "task208_direct_color_argmin", str(args.out))),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
