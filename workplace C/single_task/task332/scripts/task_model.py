from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


TASK_ID = "task332"


def solve(grid: list[list[int]]) -> list[list[int]]:
    """Recolor the unique 5 in every column of parity opposite the grid width."""
    width = len(grid[0]) if grid else 0
    output = [row[:] for row in grid]
    for row in range(len(output)):
        for column in range(width):
            if output[row][column] == 5 and (column + width) % 2 == 1:
                output[row][column] = 3
    return output


def build_onnx(output_path: Path) -> Path:
    """Build the parity mask from the count of channel-5 cells and emit one-hot output."""
    nodes = [
        helper.make_node("Slice", ["input", "crop_start", "crop_end", "crop_axes"], ["crop"]),
        helper.make_node("Gather", ["crop", "zero_index"], ["channel_zero"], axis=1),
        helper.make_node("Gather", ["crop", "five_index"], ["channel_five"], axis=1),
        helper.make_node("ReduceSum", ["channel_five", "spatial_axes"], ["width"], keepdims=0),
        helper.make_node("Mod", ["width", "two"], ["width_parity"], fmod=1),
        helper.make_node("Equal", ["width_parity", "one"], ["odd_width"]),
        helper.make_node("Where", ["odd_width", "even_columns", "odd_columns"], ["recolor_columns"]),
        helper.make_node("Mul", ["channel_five", "recolor_columns"], ["channel_three"]),
        helper.make_node("Sub", ["channel_five", "channel_three"], ["remaining_five"]),
        helper.make_node("Sub", ["channel_five", "channel_five"], ["zero_channel"]),
        helper.make_node(
            "Concat",
            [
                "channel_zero",
                "zero_channel",
                "zero_channel",
                "channel_three",
                "zero_channel",
                "remaining_five",
                "zero_channel",
                "zero_channel",
                "zero_channel",
                "zero_channel",
            ],
            ["small_output"],
            axis=1,
        ),
        helper.make_node("Pad", ["small_output", "output_pads", "zero"], ["output"], mode="constant"),
    ]
    even_columns = np.array([1.0 if column % 2 == 0 else 0.0 for column in range(30)], dtype=np.float32).reshape(1, 1, 1, 30)
    odd_columns = 1.0 - even_columns
    tensors = [
        numpy_helper.from_array(np.array([0, 0], dtype=np.int64), name="crop_start"),
        numpy_helper.from_array(np.array([3, 20], dtype=np.int64), name="crop_end"),
        numpy_helper.from_array(np.array([2, 3], dtype=np.int64), name="crop_axes"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), name="zero_index"),
        numpy_helper.from_array(np.array([5], dtype=np.int64), name="five_index"),
        numpy_helper.from_array(np.array([2, 3], dtype=np.int64), name="spatial_axes"),
        numpy_helper.from_array(np.array(2.0, dtype=np.float32), name="two"),
        numpy_helper.from_array(np.array(1.0, dtype=np.float32), name="one"),
        numpy_helper.from_array(even_columns[:, :, :, :20], name="even_columns"),
        numpy_helper.from_array(odd_columns[:, :, :, :20], name="odd_columns"),
        numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 0, 27, 10], dtype=np.int64), name="output_pads"),
        numpy_helper.from_array(np.array(0.0, dtype=np.float32), name="zero"),
    ]
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "task332_dynamic_width_parity",
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
