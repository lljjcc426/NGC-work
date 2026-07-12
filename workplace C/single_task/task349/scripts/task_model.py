from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


TASK_ID = "task349"
BASELINE = Path(
    r"E:/kagglegolf/submissions/candidates/"
    r"GOLF_20260709_101_prvsiyan_7266_72_repro/onnx/task349.onnx"
)


def _components(grid: list[list[int]], color: int) -> list[list[tuple[int, int]]]:
    height = len(grid)
    width = len(grid[0]) if height else 0
    seen: set[tuple[int, int]] = set()
    result: list[list[tuple[int, int]]] = []
    for row in range(height):
        for col in range(width):
            if grid[row][col] != color or (row, col) in seen:
                continue
            queue = deque([(row, col)])
            seen.add((row, col))
            component: list[tuple[int, int]] = []
            while queue:
                current_row, current_col = queue.popleft()
                component.append((current_row, current_col))
                for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    neighbor = current_row + delta_row, current_col + delta_col
                    neighbor_row, neighbor_col = neighbor
                    if not (0 <= neighbor_row < height and 0 <= neighbor_col < width):
                        continue
                    if neighbor in seen or grid[neighbor_row][neighbor_col] != color:
                        continue
                    seen.add(neighbor)
                    queue.append(neighbor)
            result.append(component)
    return result


def solve(grid: list[list[int]]) -> list[list[int]]:
    """Apply the task349 rectangle-halo and downward-ray rule."""
    if not grid or not grid[0]:
        return []
    height = len(grid)
    width = len(grid[0])
    output = [row[:] for row in grid]

    # Every 9 cell starts a downward ray; underfill keeps non-background cells.
    for col in range(width):
        marker_rows = [row for row in range(height) if grid[row][col] == 9]
        if not marker_rows:
            continue
        for row in range(min(marker_rows), height):
            if output[row][col] == 0:
                output[row][col] = 1

    # Generated examples extend each solid rectangle by width / 2 layers.
    for component in _components(grid, 9):
        rows = [row for row, _ in component]
        cols = [col for _, col in component]
        radius = (max(cols) - min(cols) + 1) // 2
        for row in range(max(0, min(rows) - radius), min(height, max(rows) + radius + 1)):
            for col in range(max(0, min(cols) - radius), min(width, max(cols) + radius + 1)):
                output[row][col] = max(output[row][col], 3)

    for row in range(height):
        for col in range(width):
            if grid[row][col] == 9:
                output[row][col] = 9
    return output


def _replace_initializer(model: onnx.ModelProto, name: str, value: np.ndarray) -> None:
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name == name:
            model.graph.initializer[index].CopyFrom(numpy_helper.from_array(value, name=name))
            return
    model.graph.initializer.append(numpy_helper.from_array(value, name=name))


def build_onnx(output_path: Path) -> Path:
    """Build the accepted three-channel collision-safe width/halo model."""
    from build_double_collision_merge import build

    return build(Path(output_path))
