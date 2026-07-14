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
    / "task208.onnx"
)
DEFAULT_OUT = (
    ROOT
    / "reconstruction_candidates"
    / "b_task208_compact_search_v1"
    / "task208.onnx"
)


def _add_initializer(model: onnx.ModelProto, name: str, value: np.ndarray) -> str:
    model.graph.initializer.append(numpy_helper.from_array(value, name=name))
    return name


def rewrite(model: onnx.ModelProto) -> onnx.ModelProto:
    current = onnx.ModelProto.FromString(model.SerializeToString())
    outputs = {output for node in current.graph.node for output in node.output}
    required = {"inv", "score", "k_b", "kernel", "idx14t", "zeroed", "oh_b", "oh_f"}
    if not required.issubset(outputs):
        raise RuntimeError(f"task208 graph mismatch: {sorted(required - outputs)}")

    depth10 = _add_initializer(current, "task208_depth10", np.array(10, dtype=np.int64))
    onehot_values = _add_initializer(
        current,
        "task208_onehot_values",
        np.array([0.0, 1.0], dtype=np.float32),
    )
    for index, initializer in enumerate(current.graph.initializer):
        if initializer.name == "upd0":
            current.graph.initializer[index].CopyFrom(
                numpy_helper.from_array(np.array(0, dtype=np.uint8), name="upd0")
            )
            break
    else:
        raise RuntimeError("missing upd0 initializer")

    replacements: dict[str, list[onnx.NodeProto]] = {
        "score": [
            helper.make_node(
                "Where",
                ["present", "c19", "thousand_f"],
                ["score"],
                name="task208_mask_absent_colors",
            )
        ],
        "boxidx": [
            helper.make_node(
                "ArgMin",
                ["score"],
                ["boxidx"],
                name="task208_least_present_color",
                axis=1,
                keepdims=0,
            )
        ],
        "k_b": [
            helper.make_node("Cast", ["k_c"], ["task208_k_c_u8"], name="task208_k_c_u8", to=onnx.TensorProto.UINT8),
            helper.make_node(
                "Where",
                ["k_r", "task208_k_c_u8", "zero_u8"],
                ["kernel"],
                name="task208_outer_kernel",
            ),
        ],
        "zeroed": [
            helper.make_node(
                "ScatterND",
                ["convmap", "idx4", "upd0"],
                ["zeroed"],
                name="task208_zero_first_box",
            )
        ],
        "oh_b": [
            helper.make_node(
                "Unsqueeze",
                ["boxc"],
                ["task208_boxc_111"],
                name="task208_boxc_111",
                axes=[1, 2],
            ),
            helper.make_node(
                "OneHot",
                ["task208_boxc_111", depth10, onehot_values],
                ["oh_f"],
                name="task208_box_color_onehot",
                axis=1,
            ),
        ],
    }
    remove_outputs = {"inv", "kernel", "idx14t", "oh_f"}

    rewritten: list[onnx.NodeProto] = []
    replaced: set[str] = set()
    for node in current.graph.node:
        hit = next((output for output in node.output if output in replacements), None)
        if hit is not None:
            rewritten.extend(replacements[hit])
            replaced.add(hit)
            continue
        if any(output in remove_outputs for output in node.output):
            continue
        rewritten.append(node)

    if replaced != set(replacements):
        raise RuntimeError(f"failed replacements: {sorted(set(replacements) - replaced)}")

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
    examples = json.loads((ROOT / "data" / "competition" / "task208.json").read_text())
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
        "task": 208,
        "method": "compact color selection, outer kernel, scatter index, and one-hot",
        "equivalence": _official_equivalence(args.base, args.out),
        "score": build_blend.validate_and_score((208, "task208_compact_search", str(args.out))),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
