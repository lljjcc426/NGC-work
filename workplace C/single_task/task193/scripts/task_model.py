from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


TASK_ID = "task193"
BASELINE = Path(
    r"E:/kagglegolf/submissions/candidates/"
    r"GOLF_20260709_101_prvsiyan_7266_72_repro/onnx/task193.onnx"
)


def solve(grid: list[list[int]]) -> list[list[int]]:
    """Remove rare-color cells that have fewer than two orthogonal peers."""
    if not grid or not grid[0]:
        return []
    counts = Counter(value for row in grid for value in row if value != 0)
    if not counts:
        return [row[:] for row in grid]
    rare_color = min(counts, key=lambda color: (counts[color], color))
    output = [row[:] for row in grid]
    height = len(grid)
    width = len(grid[0])
    for row in range(height):
        for col in range(width):
            if grid[row][col] != rare_color:
                continue
            direct_peers = sum(
                0 <= row + delta_row < height
                and 0 <= col + delta_col < width
                and grid[row + delta_row][col + delta_col] == rare_color
                for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1))
            )
            if direct_peers < 2:
                output[row][col] = 0
    return output


def build_onnx(output_path: Path) -> Path:
    """Factor the dense Conv while preserving its raw output exactly."""
    if not BASELINE.exists():
        raise FileNotFoundError(f"task193 baseline not found: {BASELINE}")
    baseline = onnx.load(str(BASELINE))
    initializers = {item.name: numpy_helper.to_array(item) for item in baseline.graph.initializer}
    dense_weight = initializers["W"]
    dense_bias = initializers["Bc"]
    foreground_weight = np.stack([dense_weight[color, color] for color in range(1, 10)]).reshape(9, 1, 3, 3)
    foreground_sum_weight = np.zeros((1, 1, 3, 3), dtype=np.float32)
    for row, col in ((0, 1), (1, 0), (1, 2), (2, 1)):
        foreground_sum_weight[0, 0, row, col] = -2.0
    foreground_sum_weight[0, 0, 1, 1] = 12.0
    nodes = [
        helper.make_node("Gather", ["input", "foreground_indices"], ["foreground"], axis=1),
        helper.make_node("Conv", ["foreground", "foreground_weight", "foreground_bias"], ["foreground_logits"], group=9, kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("ReduceSum", ["foreground", "channel_axis"], ["foreground_sum"], keepdims=1),
        helper.make_node("Conv", ["foreground_sum", "background_foreground_weight", "background_bias"], ["background_from_foreground"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Gather", ["input", "background_index"], ["background"], axis=1),
        helper.make_node("Mul", ["background", "background_center_weight"], ["background_center"]),
        helper.make_node("Add", ["background_from_foreground", "background_center"], ["background_logits"]),
        helper.make_node("Concat", ["background_logits", "foreground_logits"], ["output"], axis=1),
    ]
    tensors = [
        numpy_helper.from_array(np.arange(1, 10, dtype=np.int64), name="foreground_indices"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), name="background_index"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name="channel_axis"),
        numpy_helper.from_array(foreground_weight.astype(np.float32), name="foreground_weight"),
        numpy_helper.from_array(dense_bias[1:].astype(np.float32), name="foreground_bias"),
        numpy_helper.from_array(foreground_sum_weight, name="background_foreground_weight"),
        numpy_helper.from_array(np.array([dense_bias[0]], dtype=np.float32), name="background_bias"),
        numpy_helper.from_array(np.array([18.0], dtype=np.float32), name="background_center_weight"),
    ]
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "task193_exact_factor",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
            initializer=tensors,
        ),
        opset_imports=[helper.make_operatorsetid("", 13)],
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(output_path))
    return output_path
