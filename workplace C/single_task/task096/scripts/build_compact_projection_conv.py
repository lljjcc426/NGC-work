from __future__ import annotations

import csv
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


TASK = "task096"
HERE = Path(__file__).resolve()
TASK_DIR = HERE.parents[1]
REPO = HERE.parents[4]
COMMON = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"
BASE = Path(r"E:/kagglegolf/submissions/downloaded_best/v93_7273_37_user_upload/onnx/task096.onnx")
DEBUG = TASK_DIR / "debug" / "task096_compact_projection_conv.onnx"
FINAL = TASK_DIR / "onnx" / "task096_candidate.onnx"


def _replace_initializer(model: onnx.ModelProto, name: str, value: np.ndarray) -> None:
    for index, item in enumerate(model.graph.initializer):
        if item.name == name:
            model.graph.initializer[index].CopyFrom(numpy_helper.from_array(value, name=name))
            return
    model.graph.initializer.append(numpy_helper.from_array(value, name=name))


def build(output_path: Path = DEBUG) -> Path:
    model = deepcopy(onnx.load(BASE))
    nodes = list(model.graph.node)
    if len(nodes) != 113 or nodes[0].op_type != "ReduceSum" or nodes[47].op_type != "ReduceSum":
        raise RuntimeError("unexpected task096 baseline graph")

    # All task096 source grids fit in the first 19 rows/columns. Negative tail
    # pads make each depthwise Conv both crop to that support and sum the other
    # spatial axis, avoiding the baseline's two 10x30 float projections.
    row_weight = np.ones((10, 1, 1, 19), dtype=np.float32)
    col_weight = np.ones((10, 1, 19, 1), dtype=np.float32)
    _replace_initializer(model, "compact_row_weight", row_weight)
    _replace_initializer(model, "compact_col_weight", col_weight)
    _replace_initializer(model, "compact_count_axes", np.array([0, 2, 3], dtype=np.int64))
    _replace_initializer(model, "compact_row_squeeze_axes", np.array([0, 3], dtype=np.int64))
    _replace_initializer(model, "compact_col_squeeze_axes", np.array([0, 2], dtype=np.int64))

    nodes[0].CopyFrom(
        helper.make_node(
            "Conv",
            ["input", "compact_row_weight"],
            ["row_sum_compact"],
            group=10,
            kernel_shape=[1, 19],
            pads=[0, 0, -11, -11],
        )
    )
    nodes[1].input[:] = ["row_sum_compact", "compact_count_axes"]
    nodes[7].input[:] = ["row_sum_compact"]
    nodes[7].output[:] = ["row_present_compact"]
    nodes[8].input[:] = ["row_present_compact", "top_colors_1"]
    nodes[8].output[:] = ["row_present_selected_compact"]
    for attribute in nodes[8].attribute:
        if attribute.name == "axis":
            attribute.i = 1
    nodes[9].CopyFrom(
        helper.make_node(
            "Squeeze",
            ["row_present_selected_compact", "compact_row_squeeze_axes"],
            ["row_present19"],
        )
    )

    nodes[47].CopyFrom(
        helper.make_node(
            "Conv",
            ["input", "compact_col_weight"],
            ["col_sum_compact"],
            group=10,
            kernel_shape=[19, 1],
            pads=[0, 0, -11, -11],
        )
    )
    nodes[48].input[:] = ["col_sum_compact"]
    nodes[48].output[:] = ["col_present_compact"]
    nodes[49].input[:] = ["col_present_compact", "top_colors_1"]
    nodes[49].output[:] = ["col_present_selected_compact"]
    for attribute in nodes[49].attribute:
        if attribute.name == "axis":
            attribute.i = 1
    nodes[50].CopyFrom(
        helper.make_node(
            "Squeeze",
            ["col_present_selected_compact", "compact_col_squeeze_axes"],
            ["col_present19"],
        )
    )

    del model.graph.node[:]
    model.graph.node.extend(nodes)
    used = {name for node in model.graph.node for name in node.input}
    kept = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    replaced_values = {
        "row_sum_full",
        "row_present_full",
        "row_present_selected",
        "col_sum_full",
        "col_present_full",
        "col_present_selected",
    }
    kept_value_info = [item for item in model.graph.value_info if item.name not in replaced_values]
    kept_value_info.extend(
        [
            helper.make_tensor_value_info("row_sum_compact", TensorProto.FLOAT, [1, 10, 19, 1]),
            helper.make_tensor_value_info("row_present_compact", TensorProto.BOOL, [1, 10, 19, 1]),
            helper.make_tensor_value_info("row_present_selected_compact", TensorProto.BOOL, [1, 5, 19, 1]),
            helper.make_tensor_value_info("col_sum_compact", TensorProto.FLOAT, [1, 10, 1, 19]),
            helper.make_tensor_value_info("col_present_compact", TensorProto.BOOL, [1, 10, 1, 19]),
            helper.make_tensor_value_info("col_present_selected_compact", TensorProto.BOOL, [1, 5, 1, 19]),
        ]
    )
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_value_info)
    model.producer_name = "ngc_c_task096_compact_projection_conv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(output_path))
    return output_path


def main() -> None:
    sys.path.insert(0, str(COMMON))
    from c_score_common import score_onnx

    candidate = build()
    old = score_onnx(TASK, BASE, True)
    new = score_onnx(TASK, candidate, True)
    row = {
        "task": TASK,
        "method": "compact_projection_conv",
        "old_cost": old.cost,
        "new_cost": new.cost,
        "delta_cost": None if old.cost is None or new.cost is None else new.cost - old.cost,
        "old_points": old.points,
        "new_points": new.points,
        "delta_points": None if old.points is None or new.points is None else new.points - old.points,
        "examples_passed": new.examples_passed,
        "examples_checked": new.examples_checked,
        "local_valid": str(new.ok).lower(),
        "accepted": str(bool(new.ok and new.cost is not None and old.cost is not None and new.cost < old.cost)).lower(),
        "artifact_path": str(candidate),
        "error": new.error,
    }
    report = TASK_DIR / "reports" / "cost_diff_round2.csv"
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(row)
    if new.ok and new.cost is not None and old.cost is not None and new.cost < old.cost:
        FINAL.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, FINAL)
        print(FINAL)


if __name__ == "__main__":
    main()
