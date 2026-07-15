from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import build_blend  # noqa: E402
import optimize_equivalent as oe  # noqa: E402


DEFAULT_BASE = (
    ROOT
    / "reconstruction_candidates"
    / "b_task076_restore_fallback_v2"
    / "task076.onnx"
)
DEFAULT_OUT = (
    ROOT
    / "reconstruction_candidates"
    / "b_task076_even_sentinel_seed_v3"
    / "task076.onnx"
)


def rewrite(source: onnx.ModelProto) -> onnx.ModelProto:
    model = onnx.ModelProto.FromString(source.SerializeToString())
    replacements = {
        "color15_kernel": np.zeros((1, 10, 2, 2), dtype=np.float32),
        "board_bias": np.asarray([6.0], dtype=np.float32),
        "iota_u8": np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1),
    }
    replacements["color15_kernel"][0, :, 0, 0] = np.arange(10, dtype=np.float32) - 6.0
    replacements["iota_u8"][0, 5, 0, 0] = 99
    replacements["iota_u8"][0, 6, 0, 0] = 99
    replaced_initializers: set[str] = set()
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name not in replacements:
            continue
        model.graph.initializer[index].CopyFrom(
            numpy_helper.from_array(replacements[initializer.name], initializer.name)
        )
        replaced_initializers.add(initializer.name)
    if replaced_initializers != set(replacements):
        raise RuntimeError(
            f"task076 sentinel initializers not found: {sorted(set(replacements) - replaced_initializers)}"
        )
    removed = {"input_ch1", "input_ch3", "source_seed", "source_step0_mask"}
    nodes: list[onnx.NodeProto] = []
    seen: set[str] = set()

    for source_node in model.graph.node:
        output = source_node.output[0] if source_node.output else ""
        if output == "source_seed":
            nodes.extend(
                [
                    helper.make_node(
                        "BitwiseAnd",
                        ["board_code", "one_u8"],
                        ["source_step0_mask"],
                        name="source_step0_mask",
                    ),
                    helper.make_node(
                        "Cast",
                        ["source_step0_mask"],
                        ["source_seed"],
                        name="source_seed",
                        to=TensorProto.BOOL,
                    ),
                ]
            )
            seen.add(output)
            continue
        if output in {"input_ch1", "input_ch3", "source_step0_mask"}:
            seen.add(output)
            continue
        nodes.append(onnx.NodeProto.FromString(source_node.SerializeToString()))

    if seen != removed:
        raise RuntimeError(f"task076 parity seed nodes not found: {sorted(removed - seen)}")
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
        "method": "even boundary sentinel with uint8 parity source seed",
        "equivalence": compare(args.base, args.out),
        "score": build_blend.validate_and_score((76, "task076_parity_seed", str(args.out))),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
