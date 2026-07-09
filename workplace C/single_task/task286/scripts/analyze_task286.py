from __future__ import annotations

import csv
import json
from collections import Counter, deque
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TASK_DIR = SCRIPT_DIR.parent
REPO_ROOT = TASK_DIR.parents[2]
TASK_JSON = REPO_ROOT / "neurogolf_400_tasks" / "tasks" / "task286.json"
REPORT_DIR = TASK_DIR / "reports"
DEBUG_DIR = TASK_DIR / "debug"


Grid = list[list[int]]


def load_task() -> dict:
    return json.loads(TASK_JSON.read_text(encoding="utf-8"))


def shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0]) if grid else 0


def colors(grid: Grid) -> set[int]:
    return {v for row in grid for v in row}


def count_changed(inp: Grid, out: Grid) -> int:
    return sum(1 for r in range(len(inp)) for c in range(len(inp[0])) if inp[r][c] != out[r][c])


def components(grid: Grid, passable: set[int]) -> list[dict]:
    h, w = shape(grid)
    seen = [[False] * w for _ in range(h)]
    out = []
    for r in range(h):
        for c in range(w):
            if seen[r][c] or grid[r][c] not in passable:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r][c] = True
            cells = []
            color_counts: Counter[int] = Counter()
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                color_counts[grid[cr][cc]] += 1
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and grid[nr][nc] in passable:
                        seen[nr][nc] = True
                        q.append((nr, nc))
            rows = [p[0] for p in cells]
            cols = [p[1] for p in cells]
            out.append(
                {
                    "size": len(cells),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "color_counts": dict(sorted(color_counts.items())),
                    "has_marker": any(k not in {0, 8} for k in color_counts),
                }
            )
    return sorted(out, key=lambda x: x["size"], reverse=True)


def write_grid_pair(path: Path, inp: Grid, out: Grid) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("input\n")
        for row in inp:
            f.write(" ".join(map(str, row)) + "\n")
        f.write("\noutput\n")
        for row in out:
            f.write(" ".join(map(str, row)) + "\n")


def analyze() -> None:
    task = load_task()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    split_counts = {split: len(task.get(split, [])) for split in ("train", "test", "arc-gen")}
    changed_ratios = []
    marker_pairs = Counter()
    component_counts = Counter()
    output_subset = True

    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(task.get(split, [])):
            inp, out = ex["input"], ex["output"]
            h, w = shape(inp)
            in_colors = colors(inp)
            out_colors = colors(out)
            output_subset = output_subset and out_colors.issubset(in_colors)
            markers = tuple(sorted(c for c in in_colors if c not in {0, 8}))
            marker_pairs[markers] += 1
            changed = count_changed(inp, out)
            changed_ratios.append(changed / (h * w))
            comps = components(inp, in_colors - {8})
            seeded_components = [c for c in comps if c["has_marker"]]
            component_counts[(len(comps), len(seeded_components))] += 1
            rows.append(
                {
                    "split": split,
                    "index": idx,
                    "shape": f"{h}x{w}",
                    "input_colors": " ".join(map(str, sorted(in_colors))),
                    "output_colors": " ".join(map(str, sorted(out_colors))),
                    "marker_colors": " ".join(map(str, markers)),
                    "changed_cells": changed,
                    "changed_ratio": f"{changed / (h * w):.6f}",
                    "passable_components": len(comps),
                    "seeded_components": len(seeded_components),
                    "largest_component": comps[0]["size"] if comps else 0,
                    "largest_seeded_component": max((c["size"] for c in seeded_components), default=0),
                }
            )
            if split in {"train", "test"} or (split == "arc-gen" and idx < 20):
                write_grid_pair(DEBUG_DIR / f"example_{split}_{idx:03d}.txt", inp, out)

    with (REPORT_DIR / "task286_examples_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with (REPORT_DIR / "task286_analysis.md").open("w", encoding="utf-8") as f:
        f.write("# task286 Analysis\n\n")
        f.write(f"- task_json: `{TASK_JSON}`\n")
        f.write(f"- examples: train `{split_counts['train']}`, test `{split_counts['test']}`, arc-gen `{split_counts['arc-gen']}`\n")
        f.write("- baseline current_cost: `26909`\n")
        f.write("- baseline current_points: `14.799783917876258`\n")
        f.write(f"- output colors subset of input colors: `{output_subset}`\n")
        f.write(f"- changed ratio avg: `{sum(changed_ratios) / len(changed_ratios):.6f}`\n")
        f.write(f"- changed cells min/median/max: `{min(int(float(r['changed_cells'])) for r in rows)}` / ")
        sorted_changed = sorted(int(r["changed_cells"]) for r in rows)
        f.write(f"`{sorted_changed[len(sorted_changed)//2]}` / `{max(sorted_changed)}`\n\n")
        f.write("## Interpretation\n\n")
        f.write("- `8` behaves as wall/barrier color.\n")
        f.write("- `0` behaves as unfilled corridor color.\n")
        f.write("- The two non `{0,8}` colors are seed/marker colors.\n")
        f.write("- Output fills only the passable connected component containing marker cells.\n")
        f.write("- Fill color alternates by 4-neighbor graph distance from the seed markers.\n")
        f.write("- Other passable components without markers remain `0`.\n\n")
        f.write("## Marker Color Pairs\n\n")
        f.write("| marker_colors | examples |\n| --- | ---: |\n")
        for pair, count in marker_pairs.most_common():
            f.write(f"| `{' '.join(map(str, pair))}` | {count} |\n")
        f.write("\n## Component Counts\n\n")
        f.write("| passable_components | seeded_components | examples |\n| ---: | ---: | ---: |\n")
        for (pc, sc), count in component_counts.most_common():
            f.write(f"| {pc} | {sc} | {count} |\n")


if __name__ == "__main__":
    analyze()
