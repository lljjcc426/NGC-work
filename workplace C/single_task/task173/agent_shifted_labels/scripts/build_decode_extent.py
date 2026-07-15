from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve()
AGENT_DIR = HERE.parents[1]
REPO = HERE.parents[5]
COMMON = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"
DEFAULT_PARENT = (
    REPO
    / "workplace C"
    / "artifacts"
    / "full400_round36_public_source_safe37"
    / "onnx"
    / "task173.onnx"
)
DEFAULT_SOURCE = (
    REPO
    / "workplace C"
    / "artifacts"
    / "safe_dilated_crop_conv"
    / "task173.onnx"
)
DEFAULT_OUTPUT = AGENT_DIR / "debug" / "task173_decode_extent.onnx"
DEFAULT_EVIDENCE = AGENT_DIR / "debug" / "decode_extent_evidence.json"


SLICE_OUTPUTS = {"rsent25", "csent25"}
SLICE_INITIALIZERS = {"slice25_starts", "slice25_ends", "slice_axis_r"}


def replace_initializer(
    model: onnx.ModelProto, name: str, value: np.ndarray
) -> None:
    for tensor in model.graph.initializer:
        if tensor.name == name:
            tensor.CopyFrom(numpy_helper.from_array(value, name=name))
            return
    raise RuntimeError(f"missing initializer: {name}")


def build(source: Path, output: Path) -> Path:
    model = deepcopy(onnx.load(source))

    weight = next(
        numpy_helper.to_array(item)
        for item in model.graph.initializer
        if item.name == "W"
    ).copy()
    if weight.shape != (1, 10, 2, 2):
        raise RuntimeError(f"unexpected safe crop weight shape: {weight.shape}")
    weight[0, 0, 0, 0] = np.asarray(0.5, dtype=weight.dtype)
    replace_initializer(model, "W", weight)

    kept_nodes = []
    rewired_reductions = set()
    rewired_terminal = False
    for node in model.graph.node:
        if any(name in SLICE_OUTPUTS for name in node.output):
            continue
        if node.output and node.output[0] in {"vr", "vc"}:
            if node.op_type != "ReduceMax" or node.input[0] != "input":
                raise RuntimeError(f"unexpected extent reduction: {node}")
            node.input[0] = "labf"
            rewired_reductions.add(node.output[0])
        if node.output and node.output[0] == "oidx":
            if node.op_type != "Max":
                raise RuntimeError(f"unexpected terminal merge: {node}")
            node.input[1] = "rsent"
            node.input[2] = "csent"
            rewired_terminal = True
        kept_nodes.append(node)
    if rewired_reductions != {"vr", "vc"} or not rewired_terminal:
        raise RuntimeError("expected extent nodes were not rewritten")
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)

    kept_initializers = [
        item
        for item in model.graph.initializer
        if item.name not in SLICE_INITIALIZERS
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)
    del model.graph.value_info[:]

    model.graph.name = "task173_safe_decode_extent"
    model.producer_name = "ngc-task173-safe-decode-extent"
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    args = parser.parse_args()

    sys.path.insert(0, str(COMMON))
    from c_score_common import score_onnx
    from candidate_registry import operator_audit

    candidate = build(args.source, args.output)
    parent_result = score_onnx("task173", args.parent, validate_all=True)
    source_result = score_onnx("task173", args.source, validate_all=True)
    candidate_result = score_onnx("task173", candidate, validate_all=True)
    audit = operator_audit(candidate)
    evidence = {
        "task": "task173",
        "rewrite": (
            "derive 25-wide row-column extents from fractional cropped label Conv "
            "and remove two sentinel Slice nodes"
        ),
        "parent": parent_result.__dict__,
        "safe_dilated_source": source_result.__dict__,
        "candidate": candidate_result.__dict__,
        "delta_from_parent": (
            None
            if parent_result.cost is None or candidate_result.cost is None
            else parent_result.cost - candidate_result.cost
        ),
        "delta_from_safe_dilated_source": (
            None
            if source_result.cost is None or candidate_result.cost is None
            else source_result.cost - candidate_result.cost
        ),
        "operator_audit": audit,
    }
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    return 0 if candidate_result.ok and audit["runtime_compatible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
