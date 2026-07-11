from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


TASK_ID = "task372"
BASELINE = Path(
    r"E:/kagglegolf/submissions/candidates/"
    r"GOLF_20260709_101_prvsiyan_7266_72_repro/onnx/task372.onnx"
)


def solve(grid: list[list[int]]) -> list[list[int]]:
    """Overlay the two five-row panels separated by the all-5 divider."""
    divider = next(index for index, row in enumerate(grid) if all(value == 5 for value in row))
    top = grid[:divider]
    bottom = grid[divider + 1 :]
    if len(top) != len(bottom):
        raise ValueError("task372 panels must have equal heights")
    return [
        [upper if upper != 0 else lower for upper, lower in zip(top_row, bottom_row)]
        for top_row, bottom_row in zip(top, bottom)
    ]


def build_onnx(output_path: Path) -> Path:
    """Crop the fixed scene before an exact grouped panel-overlay Conv."""
    if not BASELINE.exists():
        raise FileNotFoundError(f"task372 baseline not found: {BASELINE}")
    baseline = onnx.load(BASELINE)
    initializers = {item.name: numpy_helper.to_array(item) for item in baseline.graph.initializer}
    dense_weight = initializers["W"]
    dense_bias = initializers["B"].copy()
    grouped_weight = np.stack([dense_weight[color, color] for color in range(10)]).reshape(10, 1, 7, 1)
    dense_bias[0] += 1.0
    for color in range(1, 10):
        if color != 5:
            dense_bias[color] += 1.0
    nodes = [
        helper.make_node("Slice", ["input", "crop_start", "crop_end", "crop_axes"], ["crop"]),
        helper.make_node("Conv", ["crop", "W", "B"], ["panel_output"], group=10, kernel_shape=[7, 1], pads=[0, 0, 0, 0]),
        helper.make_node("Pad", ["panel_output", "output_pads", "zero"], ["output"], mode="constant"),
    ]
    tensors = [
        numpy_helper.from_array(grouped_weight, name="W"),
        numpy_helper.from_array(dense_bias, name="B"),
        numpy_helper.from_array(np.array([0, 0], dtype=np.int64), name="crop_start"),
        numpy_helper.from_array(np.array([11, 11], dtype=np.int64), name="crop_end"),
        numpy_helper.from_array(np.array([2, 3], dtype=np.int64), name="crop_axes"),
        numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 0, 25, 19], dtype=np.int64), name="output_pads"),
        numpy_helper.from_array(np.array(0.0, dtype=np.float32), name="zero"),
    ]
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "task372_cropped_group_overlay",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
            initializer=tensors,
        ),
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, output_path)
    return output_path
