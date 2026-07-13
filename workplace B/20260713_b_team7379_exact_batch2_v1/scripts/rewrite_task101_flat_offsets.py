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
import optimize_equivalent as oe  # noqa: E402


DEFAULT_BASE = (
    ROOT
    / "team_baselines"
    / "team_submission2_20260713"
    / "submission"
    / "task101.onnx"
)
DEFAULT_OUT = (
    ROOT
    / "reconstruction_candidates"
    / "b_task101_flat_offsets_v1"
    / "task101.onnx"
)


def _add_initializer(model: onnx.ModelProto, name: str, value: np.ndarray) -> str:
    model.graph.initializer.append(numpy_helper.from_array(value, name=name))
    return name


def rewrite(model: onnx.ModelProto) -> onnx.ModelProto:
    current = onnx.ModelProto.FromString(model.SerializeToString())
    outputs = {out for node in current.graph.node for out in node.output}
    required = {
        "dT",
        "dX",
        "dY",
        "d0",
        "d1",
        "d2",
        "d3",
        "d4",
        "d5",
        "eg",
        "eh",
        "ej",
        "ek",
        "el",
        "em",
        "en",
        "eo",
    }
    if not required.issubset(outputs):
        missing = sorted(required - outputs)
        raise RuntimeError(f"task101 graph no longer matches flat-offset rewrite: {missing}")

    flat3 = _add_initializer(
        current,
        "task101_flat_offsets3",
        np.array([1, 20, 21], dtype=np.float16).reshape(1, 1, 3),
    )
    flat5 = _add_initializer(
        current,
        "task101_flat_offsets5",
        np.array([2, 22, 40, 41, 42], dtype=np.float16).reshape(1, 1, 5),
    )

    remove = {"dX", "d1", "d4", "eg", "ek", "el", "en"}
    replacements: dict[str, list[onnx.NodeProto]] = {
        "d2": [helper.make_node("Not", ["d0"], ["d2"], name="task101_valid_col3")],
        "d5": [
            helper.make_node(
                "Add",
                ["dT", flat3],
                ["d5"],
                name="task101_flat_indices3",
            )
        ],
        "el": [helper.make_node("Not", ["ej"], ["el"], name="task101_valid_col5")],
        "eo": [
            helper.make_node("Mul", ["ed", "j"], ["task101_base_row5"], name="task101_base_row5"),
            helper.make_node("Add", ["task101_base_row5", "ef"], ["task101_base_flat5"], name="task101_base_flat5"),
            helper.make_node(
                "Add",
                ["task101_base_flat5", flat5],
                ["eo"],
                name="task101_flat_indices5",
            ),
        ],
    }

    rewritten: list[onnx.NodeProto] = []
    replaced: set[str] = set()
    for node in current.graph.node:
        hit = next((out for out in node.output if out in replacements), None)
        if hit is not None:
            rewritten.extend(replacements[hit])
            replaced.add(hit)
            continue
        if any(out in remove for out in node.output):
            continue
        rewritten.append(node)

    if replaced != set(replacements):
        raise RuntimeError(f"failed to replace outputs: {sorted(set(replacements) - replaced)}")

    del current.graph.node[:]
    current.graph.node.extend(rewritten)
    del current.graph.value_info[:]
    oe.prune_dead(current)
    oe.prune_initializers(current)
    onnx.checker.check_model(current, full_check=True)
    return current


def _official_equivalence(base: Path, candidate: Path) -> dict[str, int]:
    base_session = ort.InferenceSession(str(base), providers=["CPUExecutionProvider"])
    candidate_session = ort.InferenceSession(str(candidate), providers=["CPUExecutionProvider"])
    examples = json.loads((ROOT / "data" / "competition" / "task101.json").read_text())
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(rewrite(onnx.load(args.base)), args.out)
    result = {
        "task": 101,
        "method": "precomposed flat placement offsets",
        "equivalence": _official_equivalence(args.base, args.out),
        "score": build_blend.validate_and_score((101, "task101_flat_offsets", str(args.out))),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
