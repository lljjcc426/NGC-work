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
BASE = Path(r"E:/kagglegolf/submissions/candidates/GOLF_20260709_101_prvsiyan_7266_72_repro/onnx/task072.onnx")
DEBUG = TASK_DIR / "debug" / "task072_conv_difference.onnx"
FINAL = TASK_DIR / "onnx" / "task072_candidate.onnx"


def _init(name: str, value: np.ndarray):
    return numpy_helper.from_array(value, name=name)


def build(output: Path = DEBUG) -> Path:
    weights = np.zeros((1, 10, 8, 1), dtype=np.float32)
    weights[0, 2, 0, 0] = -1.0
    weights[0, 2, 7, 0] = 1.0
    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv",
                ["input", "weights", "bias"],
                ["difference"],
                kernel_shape=[8, 1],
                pads=[0, 0, -17, -25],
            ),
            helper.make_node("Equal", ["difference", "zero_f32"], ["same"]),
            helper.make_node("Where", ["same", "same_value", "different_value"], ["small"]),
            helper.make_node("Pad", ["small", "pads"], ["output"]),
        ],
        "task072_conv_difference",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.UINT8, [1, 10, 30, 30])],
        [
            _init("weights", weights),
            _init("bias", np.zeros(1, dtype=np.float32)),
            _init("zero_f32", np.array(0.0, dtype=np.float32)),
            _init("same_value", np.array([[[[1]], [[0]], [[0]], [[0]]]], dtype=np.uint8)),
            _init("different_value", np.array([[[[0]], [[0]], [[0]], [[1]]]], dtype=np.uint8)),
            _init("pads", np.array([0, 0, 0, 0, 0, 6, 24, 25], dtype=np.int64)),
        ],
        value_info=[
            helper.make_tensor_value_info("difference", TensorProto.FLOAT, [1, 1, 6, 5]),
            helper.make_tensor_value_info("same", TensorProto.BOOL, [1, 1, 6, 5]),
            helper.make_tensor_value_info("small", TensorProto.UINT8, [1, 4, 6, 5]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.producer_name = "ngc_c_task072_conv_difference"
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(output))
    return output


def main() -> None:
    sys.path.insert(0, str(COMMON))
    from c_score_common import score_onnx

    candidate = build()
    old = score_onnx(TASK, BASE, True)
    new = score_onnx(TASK, candidate, True)
    accepted = bool(new.ok and new.cost is not None and old.cost is not None and new.cost < old.cost)
    row = {
        "task": TASK,
        "method": "single_conv_difference",
        "old_cost": old.cost,
        "new_cost": new.cost,
        "delta_cost": None if old.cost is None or new.cost is None else new.cost - old.cost,
        "old_points": old.points,
        "new_points": new.points,
        "delta_points": None if old.points is None or new.points is None else new.points - old.points,
        "examples_passed": new.examples_passed,
        "examples_checked": new.examples_checked,
        "local_valid": str(new.ok).lower(),
        "accepted": str(accepted).lower(),
        "artifact_path": str(candidate),
        "error": new.error,
    }
    report = TASK_DIR / "reports" / "cost_diff_round2.csv"
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    if accepted:
        FINAL.write_bytes(candidate.read_bytes())
    print(row)


if __name__ == "__main__":
    main()
