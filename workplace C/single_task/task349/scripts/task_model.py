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
    """Build a validated baseline-equivalent compressed ONNX.

    The two/four-channel halo encodings reduce cost more aggressively, but the
    public arc-gen set contains overlapping small rectangles that create false
    positives after channel packing. This build keeps the proven five-channel
    rule and removes the unused width-10 right-boundary check from the
    horizontal detector. Width 10 is the maximum task width, so the shortened
    11-wide detector is equivalent on the full public set and saves 5 params.
    """
    if not BASELINE.exists():
        raise FileNotFoundError(f"task349 baseline not found: {BASELINE}")
    model = onnx.load(str(BASELINE))
    nodes = list(model.graph.node)
    horizontal = next(node for node in nodes if node.output == ["h_pos_u8"])

    base_kernel = next(
        numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
        if initializer.name == "h_kernel_combined_i8"
    )
    horizontal_kernel = base_kernel[:, :, :, :11].copy()
    _replace_initializer(model, "h_kernel_combined_i8", horizontal_kernel)
    for attr in horizontal.attribute:
        if attr.name == "kernel_shape":
            attr.ints[:] = [1, 11]
        elif attr.name == "pads":
            attr.ints[:] = [0, 1, 0, 9]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(output_path))
    return output_path
