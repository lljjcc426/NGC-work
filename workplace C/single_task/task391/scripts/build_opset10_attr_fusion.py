from __future__ import annotations

import csv
import sys
from copy import deepcopy
from pathlib import Path

import onnx
import numpy as np
from onnx import helper, numpy_helper


TASK = "task391"
HERE = Path(__file__).resolve()
TASK_DIR = HERE.parents[1]
REPO = HERE.parents[4]
COMMON = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"
BASE = Path(r"E:/kagglegolf/submissions/downloaded_best/v93_7273_37_user_upload/onnx/task391.onnx")
OUT_DIR = TASK_DIR / "onnx"


def build_onnx(mode: str = "direct_onehot") -> Path:
    model = deepcopy(onnx.load(BASE))
    nodes = list(model.graph.node)
    if [node.op_type for node in nodes] != ["ReduceSum", "TopK", "Cast", "Gather", "Equal", "Pad"]:
        raise RuntimeError("unexpected task391 graph")
    # TopK indices are already int64. Gather them directly into the logical
    # [batch, three colors, width] shape and let OneHot create the channel axis.
    pick = next(item for item in model.graph.initializer if item.name == "pick")
    pick.CopyFrom(numpy_helper.from_array(np.array([[[2], [3], [4]]], dtype=np.int64), name="pick"))
    nodes[2:5] = [
        helper.make_node("Gather", ["ti", "pick"], ["sel"], axis=0, name="select_three_indices"),
        helper.make_node("Sub", ["sel", "index_one"], ["sel_zero_based"], name="zero_base_color_indices"),
        helper.make_node("OneHot", ["sel_zero_based", "onehot_depth", "onehot_values"], ["small"], axis=1, name="selected_color_onehot"),
    ]
    model.graph.initializer.extend([
        numpy_helper.from_array(np.array(1, dtype=np.int64), name="index_one"),
        numpy_helper.from_array(np.array(9, dtype=np.int64), name="onehot_depth"),
        numpy_helper.from_array(np.array([0.0, 1.0], dtype=np.float32), name="onehot_values"),
    ])
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    kept = [item for item in model.graph.initializer if item.name != "ids"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    # Drop stale inferred types for the replaced uint8 Cast/Gather/Equal path.
    del model.graph.value_info[:]
    model.graph.output[0].type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    del model.opset_import[:]
    model.opset_import.extend([helper.make_opsetid("", 13)])
    path = OUT_DIR / f"task391_{mode}.onnx"
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, path)
    return path


def main() -> None:
    sys.path.insert(0, str(COMMON))
    from c_score_common import score_onnx

    old = score_onnx(TASK, BASE, True)
    mode = "direct_onehot"
    candidate = build_onnx(mode)
    new = score_onnx(TASK, candidate, True)
    rows = [{
            "task": TASK, "method": mode, "old_cost": old.cost,
            "new_cost": new.cost, "delta_cost": None if old.cost is None or new.cost is None else new.cost - old.cost,
            "old_points": old.points, "new_points": new.points,
            "delta_points": None if old.points is None or new.points is None else new.points - old.points,
            "examples_passed": new.examples_passed, "examples_checked": new.examples_checked,
            "local_valid": str(new.ok).lower(), "accepted": str(bool(new.ok and new.cost < old.cost)).lower(),
            "artifact_path": str(candidate),
        }]
    report = TASK_DIR / "reports" / "cost_diff_round2.csv"
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(rows)


if __name__ == "__main__":
    main()
