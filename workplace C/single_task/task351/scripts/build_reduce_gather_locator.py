from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

TASK = "task351"
HERE = Path(__file__).resolve()
TASK_DIR = HERE.parents[1]
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO / "workplace C" / "neurogolf-2026-work" / "scripts"))
from c_score_common import CURRENT_BEST_ONNX_DIR, score_onnx  # noqa: E402


def ini(name: str, value) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value), name=name)


def build(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nodes = [
        helper.make_node("ReduceSum", ["input", "axis_w"], ["all_row_counts"], name="count_each_color_by_row", keepdims=0),
        helper.make_node("Gather", ["all_row_counts", "color3"], ["rowsum_3d"], name="select_marker_rows", axis=1),
        helper.make_node("Squeeze", ["rowsum_3d", "squeeze_axis"], ["rowsum"], name="squeeze_marker_rows"),
        helper.make_node("ReduceSum", ["input", "axis_h"], ["all_col_counts"], name="count_each_color_by_col", keepdims=0),
        helper.make_node("Gather", ["all_col_counts", "color3"], ["colsum_3d"], name="select_marker_cols", axis=1),
        helper.make_node("Squeeze", ["colsum_3d", "squeeze_axis"], ["colsum"], name="squeeze_marker_cols"),
        helper.make_node("ArgMax", ["rowsum"], ["row_i64"], axis=1, keepdims=0),
        helper.make_node("ArgMax", ["colsum"], ["col_i64"], axis=1, keepdims=0),
        helper.make_node("Cast", ["row_i64"], ["row"], to=TensorProto.INT32),
        helper.make_node("Cast", ["col_i64"], ["col"], to=TensorProto.INT32),
        helper.make_node("Concat", ["zero", "row", "col"], ["zero_row_col"], axis=0),
        helper.make_node("Sub", ["start_base", "zero_row_col"], ["patch_starts"]),
        helper.make_node("Sub", ["end_base", "zero_row_col"], ["patch_ends"]),
        helper.make_node("Slice", ["input", "patch_starts", "patch_ends", "patch_axes", "patch_steps"], ["patch"]),
        helper.make_node("Pad", ["patch", "out_pads"], ["output"]),
    ]
    inits = [
        ini("axis_w", np.array([3], np.int64)), ini("axis_h", np.array([2], np.int64)),
        ini("color3", np.array([3], np.int64)), ini("squeeze_axis", np.array([1], np.int64)), ini("zero", np.array([0], np.int32)),
        ini("start_base", np.array([1, 15, 15], np.int32)), ini("end_base", np.array(10, np.int32)),
        ini("patch_axes", np.array([1, 2, 3], np.int32)), ini("patch_steps", np.array([1, -1, -1], np.int32)),
        ini("out_pads", np.array([0, 1, 0, 0, 0, 0, 25, 25], np.int64)),
    ]
    graph = helper.make_graph(nodes, "task351_reduce_gather_locator",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])], inits,
        value_info=[helper.make_tensor_value_info("patch", TensorProto.FLOAT, [1, 9, 5, 5])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=8)
    onnx.checker.check_model(model); onnx.save(model, output_path); return output_path


def main() -> None:
    candidate = build(TASK_DIR / "onnx" / f"{TASK}_candidate.onnx")
    old, new = score_onnx(TASK, CURRENT_BEST_ONNX_DIR / f"{TASK}.onnx"), score_onnx(TASK, candidate)
    (TASK_DIR / "reports").mkdir(parents=True, exist_ok=True)
    with (TASK_DIR / "reports" / "cost_diff.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task","variant","passed","checked","cost","points","valid","artifact"]); w.writeheader()
        for variant, r in [("baseline", old), ("reduce_gather_locator", new)]:
            w.writerow({"task":TASK,"variant":variant,"passed":r.examples_passed,"checked":r.examples_checked,"cost":r.cost,"points":r.points,"valid":r.ok,"artifact":r.path})
    print(old); print(new)

if __name__ == "__main__": main()
