from __future__ import annotations

import argparse
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import onnx
from onnx import TensorProto, helper


TASK = "task173"
HERE = Path(__file__).resolve()
TASK_DIR = HERE.parents[1]
REPO = HERE.parents[4]
COMMON = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"
DEFAULT_PARENT = Path(
    r"E:/kagglegolf/submissions/candidates/"
    r"GOLF_20260713_SUBMISSION8_REBASE/onnx/task173.onnx"
)

CAST_TO_SOURCE = {"gs": "gf", "protof": "proto", "poutf": "pout"}
CAST_TO_TOPK_VALUE = {"gs": "pv", "protof": "sv", "poutf": "cv"}
TOPK_VALUE_SHAPES = {"pv": [19], "sv": [3], "cv": [7]}


def build_direct_uint8_topk(
    source: Path, output: Path, removed_casts: set[str]
) -> Path:
    model = deepcopy(onnx.load(str(source)))
    found = set()
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name in removed_casts:
                node.input[index] = CAST_TO_SOURCE[name]
        if node.op_type == "Cast" and node.output[0] in removed_casts:
            found.add(node.output[0])
    if found != removed_casts:
        raise RuntimeError(f"missing task173 casts: {removed_casts - found}")
    kept_nodes = [
        node
        for node in model.graph.node
        if not (node.op_type == "Cast" and node.output[0] in removed_casts)
    ]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)

    topk_values = {CAST_TO_TOPK_VALUE[name] for name in removed_casts}
    kept_value_info = []
    updated = set()
    for value_info in model.graph.value_info:
        if value_info.name in removed_casts:
            continue
        if value_info.name in topk_values:
            value_info.type.tensor_type.elem_type = TensorProto.UINT8
            updated.add(value_info.name)
        kept_value_info.append(value_info)
    for name in topk_values - updated:
        kept_value_info.append(
            helper.make_tensor_value_info(name, TensorProto.UINT8, TOPK_VALUE_SHAPES[name])
        )
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_value_info)

    model.producer_name = "ngc_task173_direct_uint8_topk"
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, str(output))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    args = parser.parse_args()
    sys.path.insert(0, str(COMMON))
    from c_score_common import score_onnx

    debug = TASK_DIR / "debug"
    candidates = {
        "position_topk": build_direct_uint8_topk(
            args.parent, debug / "task173_position_topk_u8.onnx", {"gs"}
        ),
        "prototype_topks": build_direct_uint8_topk(
            args.parent,
            debug / "task173_prototype_topks_u8.onnx",
            {"protof", "poutf"},
        ),
        "all_topks": build_direct_uint8_topk(
            args.parent,
            debug / "task173_all_topks_u8.onnx",
            {"gs", "protof", "poutf"},
        ),
    }
    parent = score_onnx(TASK, args.parent, validate_all=True)
    best = None
    for name, candidate in candidates.items():
        result = score_onnx(TASK, candidate, validate_all=True)
        record = {
            "task": TASK,
            "candidate": name,
            "valid": result.ok,
            "passed": result.examples_passed,
            "checked": result.examples_checked,
            "parent_cost": parent.cost,
            "candidate_cost": result.cost,
            "delta_cost": (
                None
                if result.cost is None or parent.cost is None
                else parent.cost - result.cost
            ),
            "sha256": result.sha256,
            "path": str(candidate),
            "error": result.error,
        }
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if (
            result.ok
            and result.cost is not None
            and parent.cost is not None
            and result.cost < parent.cost
            and (best is None or result.cost < best[0])
        ):
            best = (result.cost, candidate)
    if best is not None:
        print(
            json.dumps(
                {
                    "local_only": str(best[1]),
                    "cost": best[0],
                    "reason": "Kaggle processing rejects uint8 TopK",
                }
            )
        )


if __name__ == "__main__":
    main()
