from __future__ import annotations

import argparse
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import onnx
from onnx import TensorProto


TASK = "task233"
HERE = Path(__file__).resolve()
TASK_DIR = HERE.parents[1]
REPO = HERE.parents[4]
COMMON = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"
DEFAULT_PARENT = Path(
    r"E:/kagglegolf/submissions/candidates/"
    r"GOLF_20260713_SUBMISSION8_REBASE/onnx/task233.onnx"
)


def build_uint8_topk(source: Path, output: Path, cast_outputs: set[str]) -> Path:
    model = deepcopy(onnx.load(str(source)))
    found = set()
    for node in model.graph.node:
        if node.op_type == "Cast" and node.output[0] in cast_outputs:
            for attr in node.attribute:
                if attr.name == "to":
                    attr.i = TensorProto.UINT8
                    found.add(node.output[0])
                    break
    if found != cast_outputs:
        raise RuntimeError(f"missing task233 TopK casts: {cast_outputs - found}")
    dtype_updates = {
        "safe_name_53": TensorProto.UINT8,
        "safe_name_54": TensorProto.UINT8,
        "safe_name_106": TensorProto.UINT8,
        "safe_name_107": TensorProto.UINT8,
    }
    needed = set(cast_outputs)
    needed.update(
        {"safe_name_54" if name == "safe_name_53" else "safe_name_107" for name in cast_outputs}
    )
    updated = set()
    for value_info in model.graph.value_info:
        if value_info.name in needed:
            value_info.type.tensor_type.elem_type = dtype_updates[value_info.name]
            updated.add(value_info.name)
    if updated != needed:
        raise RuntimeError(f"missing task233 value_info: {needed - updated}")
    model.producer_name = "ngc_task233_uint8_topk"
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
        "first_topk": build_uint8_topk(
            args.parent, debug / "task233_first_topk_u8.onnx", {"safe_name_53"}
        ),
        "match_topk": build_uint8_topk(
            args.parent, debug / "task233_match_topk_u8.onnx", {"safe_name_106"}
        ),
        "both_topk": build_uint8_topk(
            args.parent,
            debug / "task233_both_topk_u8.onnx",
            {"safe_name_53", "safe_name_106"},
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
