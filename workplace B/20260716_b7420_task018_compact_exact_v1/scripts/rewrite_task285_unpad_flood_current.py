from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import build_blend  # noqa: E402
import optimize_equivalent as oe  # noqa: E402


DEFAULT_BASE = (
    ROOT
    / "public_probe_variants"
    / "team_submission5_b_work_20260716"
    / "submission"
    / "task285.onnx"
)


def rewrite(
    source: onnx.ModelProto,
    flood_steps: int,
    remove_pad: bool,
) -> onnx.ModelProto:
    if flood_steps not in {4, 5, 6}:
        raise ValueError("flood_steps must be 4, 5, or 6")
    model = onnx.ModelProto.FromString(source.SerializeToString())

    if remove_pad:
        pads = [node for node in model.graph.node if "gp" in node.output]
        if len(pads) != 1 or pads[0].op_type != "Pad" or pads[0].input[0] != "g":
            raise RuntimeError("task285 scalar-grid Pad was not found")
        rewired = 0
        for node in model.graph.node:
            for index, name in enumerate(node.input):
                if name == "gp":
                    node.input[index] = "g"
                    rewired += 1
        if rewired != 7:
            raise RuntimeError(f"unexpected task285 Pad consumer count: {rewired}")
        kept = [node for node in model.graph.node if node is not pads[0]]

        # TopK emits 33 slots even when the grid has fewer active cells. This
        # experimental path redirects zero-color slots before removing Pad.
        safe_name = "task285_safe_linear_index"
        model.graph.initializer.append(
            numpy_helper.from_array(np.array(124, dtype=np.int32), safe_name)
        )
        t_rewired = 0
        for node in kept:
            for index, name in enumerate(node.input):
                if name == "t":
                    node.input[index] = "task285_safe_t"
                    t_rewired += 1
        if t_rewired != 3:
            raise RuntimeError(f"unexpected task285 t consumer count: {t_rewired}")
        insert_at = next(index for index, node in enumerate(kept) if "c" in node.output) + 1
        kept[insert_at:insert_at] = [
            helper.make_node(
                "Greater",
                ["c", "zero_i8"],
                ["task285_active_slot"],
                name="task285_active_slot",
            ),
            helper.make_node(
                "Where",
                ["task285_active_slot", "t", safe_name],
                ["task285_safe_t"],
                name="task285_safe_t",
            ),
        ]
        del model.graph.node[:]
        model.graph.node.extend(kept)

    if flood_steps < 6:
        replaced = 0
        for node in model.graph.node:
            for index, name in enumerate(node.input):
                if name == "exact_flood_6":
                    node.input[index] = f"exact_flood_{flood_steps}"
                    replaced += 1
        if replaced != 1:
            raise RuntimeError(f"unexpected final flood consumer count: {replaced}")

    del model.graph.value_info[:]
    oe.prune_dead(model)
    oe.prune_initializers(model)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model


def official_equivalence(base: Path, candidate: Path) -> dict[str, int]:
    base_session = ort.InferenceSession(base.read_bytes(), providers=["CPUExecutionProvider"])
    candidate_session = ort.InferenceSession(candidate.read_bytes(), providers=["CPUExecutionProvider"])
    examples = json.loads((ROOT / "data" / "competition" / "task285.json").read_text())
    checked = 0
    for split in ("train", "test", "arc-gen"):
        for example in examples.get(split, []):
            pair = build_blend.convert_to_numpy(example)
            if pair is None:
                continue
            expected = base_session.run(["output"], {"input": pair["input"]})[0]
            actual = candidate_session.run(["output"], {"input": pair["input"]})[0]
            if not np.array_equal(expected > 0, actual > 0):
                raise RuntimeError(f"official equivalence failed in {split} example {checked}")
            checked += 1
    return {"checked": checked, "matched": checked}


def stress_test(candidate: Path, examples: int, seed: int) -> dict[str, int]:
    sys.path.insert(0, str(ROOT / "third_party" / "ARC-GEN"))
    from tasks import task_b775ac94  # type: ignore  # noqa: PLC0415

    session = ort.InferenceSession(candidate.read_bytes(), providers=["CPUExecutionProvider"])
    random.seed(seed)
    for index in range(examples):
        example = task_b775ac94.generate()
        pair = build_blend.convert_to_numpy(example)
        if pair is None:
            raise RuntimeError("generator produced an unsupported grid")
        actual = session.run(["output"], {"input": pair["input"]})[0]
        if not np.array_equal(actual > 0, pair["output"] > 0):
            raise RuntimeError(f"fresh generator equivalence failed at example {index}")
    return {"checked": examples, "matched": examples, "seed": seed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--flood-steps", type=int, default=6)
    parser.add_argument("--remove-pad", action="store_true")
    parser.add_argument("--stress", type=int, default=0)
    parser.add_argument("--seed", type=int, default=285_20260716)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(rewrite(onnx.load(args.base), args.flood_steps, args.remove_pad), args.out)
    result = {
        "task": 285,
        "method": (
            f"remove_pad={args.remove_pad}, flood steps={args.flood_steps}"
        ),
        "equivalence": official_equivalence(args.base, args.out),
        "stress": stress_test(args.out, args.stress, args.seed) if args.stress else None,
        "score": build_blend.validate_and_score((285, "task285_unpad_flood", str(args.out))),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
