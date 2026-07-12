from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


TASK = "task072"
HERE = Path(__file__).resolve()
TASK_DIR = HERE.parents[1]
REPO = HERE.parents[4]
COMMON = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"
BASE = Path(r"E:/kagglegolf/submissions/candidates/GOLF_20260711_097_v96_plus_task132_task046/onnx/task072.onnx")
OUT = TASK_DIR / "onnx" / "task072_xor.onnx"


def init(name: str, value: np.ndarray):
    return numpy_helper.from_array(value, name)


def build_onnx(path: Path = OUT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nodes = [
        helper.make_node("Slice", ["input", "top_s", "top_e", "axes"], ["top2"]),
        helper.make_node("Slice", ["input", "bot_s", "bot_e", "axes"], ["bot2"]),
        helper.make_node("Equal", ["top2", "bot2"], ["same"]),
        helper.make_node("Not", ["same"], ["different"]),
        helper.make_node("And", ["same", "different"], ["zero"]),
        helper.make_node("Concat", ["same", "zero", "zero", "different"], ["small"], axis=1),
        helper.make_node("Pad", ["small", "pads", "", "pad_axes"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes,
        "task072_xor_rule",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])],
        [
            init("top_s", np.array([2, 0, 0], np.int64)),
            init("top_e", np.array([3, 6, 5], np.int64)),
            init("bot_s", np.array([2, 7, 0], np.int64)),
            init("bot_e", np.array([3, 13, 5], np.int64)),
            init("axes", np.array([1, 2, 3], np.int64)),
            init("pads", np.array([0, 0, 0, 6, 24, 25], np.int64)),
            init("pad_axes", np.array([1, 2, 3], np.int64)),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


def main() -> None:
    sys.path.insert(0, str(COMMON))
    from c_score_common import score_onnx

    candidate = build_onnx()
    old = score_onnx(TASK, BASE, True)
    new = score_onnx(TASK, candidate, True)
    rows = [
        {"model": "baseline", "passed": old.examples_passed, "checked": old.examples_checked, "memory": old.memory, "params": old.params, "cost": old.cost, "ok": old.ok, "artifact": str(BASE)},
        {"model": "xor_rule", "passed": new.examples_passed, "checked": new.examples_checked, "memory": new.memory, "params": new.params, "cost": new.cost, "ok": new.ok, "artifact": str(candidate)},
    ]
    report = TASK_DIR / "reports" / "cost_diff.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(rows)


if __name__ == "__main__":
    main()
