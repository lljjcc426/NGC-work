from __future__ import annotations

import argparse
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import onnx
from onnx import TensorProto, helper


TASK = "task366"
HERE = Path(__file__).resolve()
TASK_DIR = HERE.parents[1]
REPO = HERE.parents[4]
COMMON = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"
DEFAULT_PARENT = Path(
    r"E:/kagglegolf/submissions/candidates/"
    r"GOLF_20260713_SUBMISSION8_REBASE/onnx/task366.onnx"
)
CAST_TO_TOPK = {
    "f_cf153": ("f_ctopk154", [1, 3]),
    "f_fdf202": ("f_ftopk203", [1, 6]),
    "f_ndf247": ("f_ntopk248", [1, 6]),
}


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
        raise RuntimeError(f"missing task366 casts: {cast_outputs - found}")

    topk_values = {CAST_TO_TOPK[name][0] for name in cast_outputs}
    for node in model.graph.node:
        if node.op_type == "Greater" and node.input[0] in topk_values:
            source_name = node.input[0]
            del node.input[:]
            node.input.append(source_name)
            node.op_type = "Cast"
            del node.attribute[:]
            node.attribute.extend([helper.make_attribute("to", TensorProto.BOOL)])

    changed = set(cast_outputs)
    changed.update(topk_values)
    known = set()
    for value_info in model.graph.value_info:
        if value_info.name in changed:
            value_info.type.tensor_type.elem_type = TensorProto.UINT8
            known.add(value_info.name)
    for name in changed - known:
        if name in CAST_TO_TOPK:
            shape = [1, 255]
        else:
            shape = next(shape for value, shape in CAST_TO_TOPK.values() if value == name)
        model.graph.value_info.append(
            helper.make_tensor_value_info(name, TensorProto.UINT8, shape)
        )
    model.producer_name = "ngc_task366_uint8_topk"
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
        "corner_topk": build_uint8_topk(
            args.parent, debug / "task366_corner_topk_u8.onnx", {"f_cf153"}
        ),
        "feature_topks": build_uint8_topk(
            args.parent,
            debug / "task366_feature_topks_u8.onnx",
            {"f_fdf202", "f_ndf247"},
        ),
        "all_topks": build_uint8_topk(
            args.parent,
            debug / "task366_all_topks_u8.onnx",
            set(CAST_TO_TOPK),
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
