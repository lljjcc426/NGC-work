from __future__ import annotations

import csv
import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper, numpy_helper
import numpy as np


TASK = "task194"
HERE = Path(__file__).resolve()
TASK_DIR = HERE.parents[1]
REPO = HERE.parents[4]
COMMON_DIR = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"
sys.path.insert(0, str(COMMON_DIR))
from c_score_common import CURRENT_BEST_ONNX_DIR, score_onnx  # noqa: E402


def init(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(value, name=name)


def build(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nodes = [
        helper.make_node("Slice", ["input", "crop_starts", "crop_ends", "crop_axes"], ["patch"], name="crop_3x3"),
        helper.make_node("Slice", ["patch", "start", "end", "axes", "step_neg"], ["flip_w"], name="flip_width"),
        helper.make_node("Transpose", ["flip_w"], ["r90"], name="rotate_90", perm=[0, 1, 3, 2]),
        helper.make_node("Transpose", ["patch"], ["transpose"], name="transpose_hw", perm=[0, 1, 3, 2]),
        helper.make_node("Slice", ["transpose", "start", "end", "axes", "step_neg"], ["r270"], name="rotate_270"),
        helper.make_node("Slice", ["patch", "start2", "end2", "axes2", "steps2"], ["r180"], name="rotate_180"),
        helper.make_node("Concat", ["patch", "r270"], ["top"], name="top_half", axis=3),
        helper.make_node("Concat", ["r90", "r180"], ["bottom"], name="bottom_half", axis=3),
        helper.make_node("Concat", ["top", "bottom"], ["quadrants"], name="quadrant_output", axis=2),
        helper.make_node("Pad", ["quadrants", "pads"], ["output"], name="pad_30x30"),
    ]
    inits = [
        init("start", np.array([-1], np.int64)),
        init("end", np.array([-4], np.int64)),
        init("axes", np.array([3], np.int64)),
        init("step_neg", np.array([-1], np.int64)),
        init("start2", np.array([-1, -1], np.int64)),
        init("end2", np.array([-4, -4], np.int64)),
        init("axes2", np.array([2, 3], np.int64)),
        init("steps2", np.array([-1, -1], np.int64)),
        init("crop_starts", np.array([0, 0], np.int64)),
        init("crop_ends", np.array([3, 3], np.int64)),
        init("crop_axes", np.array([2, 3], np.int64)),
        init("pads", np.array([0, 0, 0, 0, 0, 0, 24, 24], np.int64)),
    ]
    graph = helper.make_graph(
        nodes,
        "task194_rotation_quadrants",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    return output_path


def main() -> None:
    candidate = build(TASK_DIR / "onnx" / f"{TASK}_candidate.onnx")
    old = score_onnx(TASK, CURRENT_BEST_ONNX_DIR / f"{TASK}.onnx")
    new = score_onnx(TASK, candidate)
    report = TASK_DIR / "reports" / "cost_diff.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "variant", "passed", "checked", "cost", "points", "valid", "artifact"])
        writer.writeheader()
        writer.writerow({"task": TASK, "variant": "baseline", "passed": old.examples_passed, "checked": old.examples_checked, "cost": old.cost, "points": old.points, "valid": old.ok, "artifact": old.path})
        writer.writerow({"task": TASK, "variant": "rotation_quadrants", "passed": new.examples_passed, "checked": new.examples_checked, "cost": new.cost, "points": new.points, "valid": new.ok, "artifact": new.path})
    print(old)
    print(new)


if __name__ == "__main__":
    main()
