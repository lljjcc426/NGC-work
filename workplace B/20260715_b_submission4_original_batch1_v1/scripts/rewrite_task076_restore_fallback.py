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


DEFAULT_BASE = ROOT / "team_baselines" / "team_submission4_20260715" / "extracted" / "task076.onnx"
DEFAULT_OUT = (
    ROOT
    / "reconstruction_candidates"
    / "b_task076_restore_fallback_v2"
    / "task076.onnx"
)


def rewrite(source: onnx.ModelProto) -> onnx.ModelProto:
    model = onnx.ModelProto.FromString(source.SerializeToString())
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(np.asarray(0, dtype=np.int32), "task076_zero_i32"),
            numpy_helper.from_array(np.asarray(224, dtype=np.int32), "task076_last_i32"),
        ]
    )
    nodes: list[onnx.NodeProto] = []
    changed: set[str] = set()

    for source_node in model.graph.node:
        node = onnx.NodeProto.FromString(source_node.SerializeToString())
        output = node.output[0] if node.output else ""

        if output == "hidden_indices":
            if node.op_type != "Where" or node.input[2] != "board_sentinel_i32":
                raise RuntimeError("unexpected task076 hidden index fallback")
            nodes.append(
                helper.make_node(
                    "Clip",
                    ["selected_safe_indices", "task076_zero_i32", "task076_last_i32"],
                    ["hidden_indices"],
                    name="hidden_indices",
                )
            )
            changed.add(output)
            continue

        if output == "hidden_update_code":
            if node.op_type != "Where" or node.input[2] != "zero_u8":
                raise RuntimeError("unexpected task076 hidden update fallback")
            nodes.append(
                helper.make_node(
                    "Gather",
                    ["board_code_flat", "hidden_indices"],
                    ["task076_original_hidden_values"],
                    name="task076_original_hidden_values",
                    axis=0,
                )
            )
            node.input[2] = "task076_original_hidden_values"
            nodes.append(node)
            changed.add(output)
            continue

        if output == "board_base226":
            if node.op_type != "Pad" or node.input[0] != "board_code_flat":
                raise RuntimeError("unexpected task076 sentinel board")
            changed.add(output)
            continue

        if output == "cm_scatter226":
            if node.op_type != "ScatterElements" or node.input[0] != "board_base226":
                raise RuntimeError("unexpected task076 board scatter")
            node.input[0] = "board_code_flat"
            node.output[0] = "cm_flat225"
            node.name = "cm_flat225"
            node.attribute.append(helper.make_attribute("reduction", "max"))
            nodes.append(node)
            changed.add(output)
            continue

        if output == "cm_flat225":
            if node.op_type != "Slice" or node.input[0] != "cm_scatter226":
                raise RuntimeError("unexpected task076 sentinel removal slice")
            changed.add(output)
            continue

        nodes.append(node)

    expected = {
        "hidden_indices",
        "hidden_update_code",
        "board_base226",
        "cm_scatter226",
        "cm_flat225",
    }
    if changed != expected:
        raise RuntimeError(f"task076 fallback paths not found: {sorted(expected - changed)}")
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
        "method": "source-restoring invalid writes on the 225-cell board",
        "equivalence": compare(args.base, args.out),
        "score": build_blend.validate_and_score((76, "task076_restore_fallback", str(args.out))),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
