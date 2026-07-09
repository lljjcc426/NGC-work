from __future__ import annotations

import csv
import json
from collections import deque
from pathlib import Path
from typing import Callable


SCRIPT_DIR = Path(__file__).resolve().parent
TASK_DIR = SCRIPT_DIR.parent
REPO_ROOT = TASK_DIR.parents[2]
TASK_JSON = REPO_ROOT / "neurogolf_400_tasks" / "tasks" / "task286.json"
REPORT_DIR = TASK_DIR / "reports"
FAILED_DIR = TASK_DIR / "debug" / "failed_examples"


Grid = list[list[int]]


def load_task() -> dict:
    return json.loads(TASK_JSON.read_text(encoding="utf-8"))


def clone_grid(grid: Grid) -> Grid:
    return [row[:] for row in grid]


def colors(grid: Grid) -> set[int]:
    return {v for row in grid for v in row}


def count_mismatch(a: Grid, b: Grid) -> int:
    return sum(1 for r in range(len(a)) for c in range(len(a[0])) if a[r][c] != b[r][c])


def write_grid(path: Path, title: str, grid: Grid) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(title + "\n")
        for row in grid:
            f.write(" ".join(str(v) for v in row) + "\n")


def solve_seeded_checker_fill(grid: Grid) -> Grid:
    """Fill the marker-connected 0-corridor component with alternating marker colors."""
    h = len(grid)
    w = len(grid[0]) if h else 0
    wall = 8
    empty = 0
    marker_colors = sorted(c for c in colors(grid) if c not in {empty, wall})
    if len(marker_colors) != 2:
        return clone_grid(grid)

    first, second = marker_colors
    other = {first: second, second: first}
    out = clone_grid(grid)
    assigned: dict[tuple[int, int], int] = {}
    queue: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in range(w):
            if grid[r][c] in other:
                assigned[(r, c)] = grid[r][c]
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        next_color = other[assigned[(r, c)]]
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if grid[nr][nc] == wall or (nr, nc) in assigned:
                continue
            assigned[(nr, nc)] = next_color
            queue.append((nr, nc))

    for (r, c), value in assigned.items():
        if out[r][c] == empty:
            out[r][c] = value
    return out


def solve_local_line_extension(grid: Grid) -> Grid:
    """Narrow failed hypothesis: extend marker colors only along straight corridors."""
    out = clone_grid(grid)
    h = len(grid)
    w = len(grid[0]) if h else 0
    marker_colors = sorted(c for c in colors(grid) if c not in {0, 8})
    if len(marker_colors) != 2:
        return out
    other = {marker_colors[0]: marker_colors[1], marker_colors[1]: marker_colors[0]}
    for r in range(h):
        for c in range(w):
            if grid[r][c] not in other:
                continue
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                color = other[grid[r][c]]
                nr, nc = r + dr, c + dc
                while 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == 0:
                    out[nr][nc] = color
                    color = other[color]
                    nr += dr
                    nc += dc
    return out


def solve_component_plain_fill(grid: Grid) -> Grid:
    """Narrow failed hypothesis: fill marker component with one dominant marker color."""
    out = clone_grid(grid)
    h = len(grid)
    w = len(grid[0]) if h else 0
    marker_colors = sorted(c for c in colors(grid) if c not in {0, 8})
    if len(marker_colors) != 2:
        return out
    fill_color = marker_colors[0]
    seen = [[False] * w for _ in range(h)]
    for sr in range(h):
        for sc in range(w):
            if seen[sr][sc] or grid[sr][sc] == 8:
                continue
            q: deque[tuple[int, int]] = deque([(sr, sc)])
            cells = []
            has_marker = False
            seen[sr][sc] = True
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                has_marker = has_marker or grid[r][c] in marker_colors
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and grid[nr][nc] != 8:
                        seen[nr][nc] = True
                        q.append((nr, nc))
            if has_marker:
                for r, c in cells:
                    if out[r][c] == 0:
                        out[r][c] = fill_color
    return out


def solve(grid: Grid) -> Grid:
    return solve_seeded_checker_fill(grid)


def iter_examples(task: dict):
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(task.get(split, [])):
            yield split, idx, example


def validate_solver(name: str, solver: Callable[[Grid], Grid], task: dict) -> list[dict]:
    rows = []
    for split, idx, ex in iter_examples(task):
        inp, expected = ex["input"], ex["output"]
        pred = solver(inp)
        mismatch = count_mismatch(pred, expected)
        if mismatch:
            base = FAILED_DIR / name / f"{split}_{idx:03d}"
            write_grid(base / "input.txt", "input", inp)
            write_grid(base / "expected.txt", "expected", expected)
            write_grid(base / "predicted.txt", "predicted", pred)
        rows.append(
            {
                "split": split,
                "index": idx,
                "shape": f"{len(inp)}x{len(inp[0])}",
                "passed": mismatch == 0,
                "changed_cells_expected": count_mismatch(inp, expected),
                "changed_cells_predicted": count_mismatch(inp, pred),
                "mismatch_count": mismatch,
                "notes": "",
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict[str, int]:
    out = {}
    for split in ("train", "test", "arc-gen"):
        split_rows = [r for r in rows if r["split"] == split]
        out[f"{split}_total"] = len(split_rows)
        out[f"{split}_passed"] = sum(1 for r in split_rows if r["passed"])
    out["total"] = len(rows)
    out["passed"] = sum(1 for r in rows if r["passed"])
    return out


def validate_all() -> dict:
    task = load_task()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    hypotheses = [
        (
            "h1_seeded_checker_fill",
            solve_seeded_checker_fill,
            "Multi-source BFS from marker cells through non-wall cells; alternate two marker colors by graph distance.",
        ),
        (
            "h2_straight_line_extension",
            solve_local_line_extension,
            "Only extend colors along straight corridors from each marker; fails at turns/branches.",
        ),
        (
            "h3_component_plain_fill",
            solve_component_plain_fill,
            "Fill the marker-connected component with one marker color; fails because parity matters.",
        ),
    ]
    final_rows: list[dict] = []
    summaries = []
    for name, fn, desc in hypotheses:
        rows = validate_solver(name, fn, task)
        summary = summarize(rows)
        summaries.append((name, desc, summary))
        if name == "h1_seeded_checker_fill":
            final_rows = rows

    fieldnames = [
        "split",
        "index",
        "shape",
        "passed",
        "changed_cells_expected",
        "changed_cells_predicted",
        "mismatch_count",
        "notes",
    ]
    with (REPORT_DIR / "rule_validation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    final_summary = summarize(final_rows)
    with (REPORT_DIR / "rule_validation.md").open("w", encoding="utf-8") as f:
        f.write("# task286 Rule Validation\n\n")
        f.write("Solver: `h1_seeded_checker_fill` / `solve()`\n\n")
        f.write("| split | passed | total |\n| --- | ---: | ---: |\n")
        for split in ("train", "test", "arc-gen"):
            f.write(f"| {split} | {final_summary[f'{split}_passed']} | {final_summary[f'{split}_total']} |\n")
        f.write(f"| all | {final_summary['passed']} | {final_summary['total']} |\n\n")
        f.write("The rule is input-only: treat `8` as wall, `0` as corridor, detect the two marker colors, and fill only marker-connected corridors with alternating marker colors by 4-neighbor graph distance.\n")

    with (REPORT_DIR / "rule_hypotheses.md").open("w", encoding="utf-8") as f:
        f.write("# task286 Rule Hypotheses\n\n")
        for name, desc, summary in summaries:
            f.write(f"## {name}\n\n")
            f.write(f"- Description: {desc}\n")
            f.write(f"- Passed: {summary['passed']} / {summary['total']}\n")
            f.write(
                f"- Split pass counts: train {summary['train_passed']}/{summary['train_total']}, "
                f"test {summary['test_passed']}/{summary['test_total']}, "
                f"arc-gen {summary['arc-gen_passed']}/{summary['arc-gen_total']}\n\n"
            )
    return final_summary


if __name__ == "__main__":
    print(json.dumps(validate_all(), indent=2, sort_keys=True))
