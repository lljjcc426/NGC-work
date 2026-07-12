from __future__ import annotations

from collections import Counter
from pathlib import Path

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
    """Build the accepted one-node 4x4 depthwise hard-margin model."""
    from build_depthwise_4x4_lp import build

    return build(Path(output_path))
