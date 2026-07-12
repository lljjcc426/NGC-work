from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


TASK = "task146"
HERE = Path(__file__).resolve()
TASK_DIR = HERE.parents[1]
REPO = HERE.parents[4]
COMMON = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"
BASE = Path(r"E:/kagglegolf/submissions/candidates/GOLF_20260711_097_v96_plus_task132_task046/onnx/task146.onnx")
OUT = TASK_DIR / "onnx" / "task146_asymmetry.onnx"


def init(name: str, value: np.ndarray):
    return numpy_helper.from_array(value, name)


def build_onnx(path: Path = OUT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    upper, lower = [], []
    for block in range(3):
        r = block * 3
        upper += [(r + 0) * 3 + 1, (r + 0) * 3 + 2, (r + 1) * 3 + 2]
        lower += [(r + 1) * 3 + 0, (r + 2) * 3 + 0, (r + 2) * 3 + 1]
    nodes = [
        helper.make_node("Slice", ["input", "area_s", "area_e", "area_axes"], ["area"]),
        helper.make_node("Reshape", ["area", "flat_shape"], ["flat"]),
        helper.make_node("Gather", ["flat", "upper_idx"], ["upper"], axis=2),
        helper.make_node("Gather", ["flat", "lower_idx"], ["lower"], axis=2),
        helper.make_node("Equal", ["upper", "lower"], ["pair_equal"]),
        helper.make_node("Cast", ["pair_equal"], ["pair_i"], to=TensorProto.INT32),
        helper.make_node("Reshape", ["pair_i", "pair_shape"], ["pairs"]),
        helper.make_node("ReduceMin", ["pairs", "reduce_axes"], ["symmetric"], keepdims=0),
        helper.make_node("Equal", ["symmetric", "zero_i"], ["asymmetric"]),
        helper.make_node("Cast", ["asymmetric"], ["asym_i"], to=TensorProto.INT32),
        helper.make_node("ArgMax", ["asym_i"], ["block"], axis=0, keepdims=0),
        helper.make_node("Mul", ["block", "three"], ["row0"]),
        helper.make_node("Add", ["row0", "rows"], ["row_idx"]),
        helper.make_node("Gather", ["input", "row_idx"], ["rows3"], axis=2),
        helper.make_node("Slice", ["rows3", "col_s", "col_e", "col_axis"], ["small"]),
        helper.make_node("Pad", ["small", "pads", "", "pad_axes"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes, "task146_unique_asymmetric_block",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        [
            init("area_s", np.array([0, 0], np.int64)), init("area_e", np.array([9, 3], np.int64)), init("area_axes", np.array([2, 3], np.int64)),
            init("flat_shape", np.array([1, 10, 27], np.int64)),
            init("upper_idx", np.array(upper, np.int64)), init("lower_idx", np.array(lower, np.int64)),
            init("pair_shape", np.array([1, 10, 3, 3], np.int64)), init("reduce_axes", np.array([0, 1, 3], np.int64)),
            init("zero_i", np.array(0, np.int32)), init("three", np.array(3, np.int64)), init("rows", np.array([0, 1, 2], np.int64)),
            init("col_s", np.array([0], np.int64)), init("col_e", np.array([3], np.int64)), init("col_axis", np.array([3], np.int64)),
            init("pads", np.array([0, 0, 27, 27], np.int64)), init("pad_axes", np.array([2, 3], np.int64)),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model); onnx.save(model, path); return path


def main() -> None:
    sys.path.insert(0, str(COMMON)); from c_score_common import score_onnx
    candidate = build_onnx(); old = score_onnx(TASK, BASE, True); new = score_onnx(TASK, candidate, True)
    rows = [
        {"model":"baseline","passed":old.examples_passed,"checked":old.examples_checked,"memory":old.memory,"params":old.params,"cost":old.cost,"ok":old.ok,"artifact":str(BASE)},
        {"model":"asymmetry_rule","passed":new.examples_passed,"checked":new.examples_checked,"memory":new.memory,"params":new.params,"cost":new.cost,"ok":new.ok,"artifact":str(candidate)},
    ]
    p=TASK_DIR/'reports'/'cost_diff.csv'; p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    print(rows)


if __name__ == '__main__': main()
