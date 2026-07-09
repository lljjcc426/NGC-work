from __future__ import annotations

import csv
import json
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
TASK_DIR = SCRIPT_DIR.parent
REPO_ROOT = TASK_DIR.parents[2]
TASK_JSON = REPO_ROOT / "neurogolf_400_tasks" / "tasks" / "task158.json"
REPORT_DIR = TASK_DIR / "reports"
DEBUG_DIR = TASK_DIR / "debug"
FAILED_DIR = DEBUG_DIR / "failed_examples"


Grid = list[list[int]]


@dataclass(frozen=True)
class Component:
    color: int
    cells: tuple[tuple[int, int], ...]
    bbox: tuple[int, int, int, int]
    square_size: int | None


@dataclass(frozen=True)
class SourceCandidate:
    top: int
    left: int
    patch: tuple[tuple[int, ...], ...]
    bg: int
    fill_color: int
    endpoint_a: int
    endpoint_b: int
    endpoint_a_pos: tuple[int, int]
    endpoint_b_pos: tuple[int, int]
    score: tuple[int, int, int]


@dataclass(frozen=True)
class Transform:
    name: str
    patch: tuple[tuple[int, ...], ...]


def load_task() -> dict:
    with TASK_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)


def clone_grid(grid: Grid) -> Grid:
    return [row[:] for row in grid]


def shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0]) if grid else 0


def dominant_color(grid: Grid) -> int:
    counts = Counter(v for row in grid for v in row)
    return counts.most_common(1)[0][0]


def cell_counts(grid: Grid) -> Counter[int]:
    return Counter(v for row in grid for v in row)


def in_bbox(cell: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def components(grid: Grid, colors: set[int] | None = None) -> list[Component]:
    h, w = shape(grid)
    seen = [[False] * w for _ in range(h)]
    out: list[Component] = []
    for r in range(h):
        for c in range(w):
            color = grid[r][c]
            if seen[r][c] or (colors is not None and color not in colors):
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r][c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and grid[nr][nc] == color:
                        seen[nr][nc] = True
                        q.append((nr, nc))
            rows = [p[0] for p in cells]
            cols = [p[1] for p in cells]
            bbox = (min(rows), min(cols), max(rows), max(cols))
            bh = bbox[2] - bbox[0] + 1
            bw = bbox[3] - bbox[1] + 1
            square_size = bh if bh == bw and len(cells) == bh * bw else None
            out.append(Component(color, tuple(sorted(cells)), bbox, square_size))
    return out


def patch_at(grid: Grid, r: int, c: int, size: int = 3) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(grid[r + rr][c + cc] for cc in range(size)) for rr in range(size))


def unique_transforms(patch: tuple[tuple[int, ...], ...]) -> list[Transform]:
    def rot90(m: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
        n = len(m)
        return tuple(tuple(m[n - 1 - r][c] for r in range(n)) for c in range(n))

    def flip_h(m: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(reversed(row)) for row in m)

    transforms: list[tuple[str, tuple[tuple[int, ...], ...]]] = []
    cur = patch
    for i in range(4):
        transforms.append((f"rot{i * 90}", cur))
        transforms.append((f"rot{i * 90}_flip_h", flip_h(cur)))
        cur = rot90(cur)

    seen: set[tuple[tuple[int, ...], ...]] = set()
    out: list[Transform] = []
    for name, mat in transforms:
        if mat not in seen:
            seen.add(mat)
            out.append(Transform(name, mat))
    return out


def positions_of(patch: tuple[tuple[int, ...], ...], color: int) -> list[tuple[int, int]]:
    return [(r, c) for r, row in enumerate(patch) for c, value in enumerate(row) if value == color]


def source_candidates(grid: Grid, require_target_pair: bool = True) -> list[SourceCandidate]:
    h, w = shape(grid)
    bg = dominant_color(grid)
    all_counts = cell_counts(grid)
    candidates: list[SourceCandidate] = []

    for r in range(h - 2):
        for c in range(w - 2):
            patch = patch_at(grid, r, c)
            fg = [(rr, cc, patch[rr][cc]) for rr in range(3) for cc in range(3) if patch[rr][cc] != bg]
            counts = Counter(value for _, _, value in fg)
            if len(counts) != 3:
                continue

            fill_color, fill_count = counts.most_common(1)[0]
            endpoint_colors = [color for color, count in counts.items() if color != fill_color and count == 1]
            if fill_count < 3 or len(endpoint_colors) != 2:
                continue

            endpoint_positions = {color: positions_of(patch, color)[0] for color in endpoint_colors}
            pos_a = endpoint_positions[endpoint_colors[0]]
            pos_b = endpoint_positions[endpoint_colors[1]]
            corners = {(0, 0), (0, 2), (2, 0), (2, 2)}
            if pos_a not in corners or pos_b not in corners:
                continue
            if pos_a[0] + pos_b[0] != 2 or pos_a[1] + pos_b[1] != 2:
                continue

            source_bbox = (r, c, r + 2, c + 2)
            pair_count, overlay_cells = estimate_overlay_capacity(
                grid, source_bbox, patch, bg, endpoint_colors[0], endpoint_colors[1]
            )
            if require_target_pair and pair_count == 0:
                continue

            outside_count = all_counts[endpoint_colors[0]] + all_counts[endpoint_colors[1]] - 2
            score = (pair_count, overlay_cells, outside_count)
            candidates.append(
                SourceCandidate(
                    top=r,
                    left=c,
                    patch=patch,
                    bg=bg,
                    fill_color=fill_color,
                    endpoint_a=endpoint_colors[0],
                    endpoint_b=endpoint_colors[1],
                    endpoint_a_pos=pos_a,
                    endpoint_b_pos=pos_b,
                    score=score,
                )
            )
    return sorted(candidates, key=lambda item: item.score, reverse=True)


def estimate_overlay_capacity(
    grid: Grid,
    source_bbox: tuple[int, int, int, int],
    patch: tuple[tuple[int, ...], ...],
    bg: int,
    endpoint_a: int,
    endpoint_b: int,
) -> tuple[int, int]:
    placements = find_placements(grid, source_bbox, patch, bg, endpoint_a, endpoint_b)
    overlay_cells = 0
    for placement in placements:
        overlay_cells += placement["overlay_cells"]
    return len(placements), overlay_cells


def target_components_by_color(
    grid: Grid, endpoint_colors: tuple[int, int], source_bbox: tuple[int, int, int, int]
) -> dict[int, list[Component]]:
    comps = components(grid, set(endpoint_colors))
    by_color = {endpoint_colors[0]: [], endpoint_colors[1]: []}
    for comp in comps:
        if comp.square_size is None:
            continue
        if any(in_bbox(cell, source_bbox) for cell in comp.cells):
            continue
        by_color[comp.color].append(comp)
    return by_color


def find_placements(
    grid: Grid,
    source_bbox: tuple[int, int, int, int],
    patch: tuple[tuple[int, ...], ...],
    bg: int,
    endpoint_a: int,
    endpoint_b: int,
) -> list[dict]:
    h, w = shape(grid)
    by_color = target_components_by_color(grid, (endpoint_a, endpoint_b), source_bbox)
    placements: list[dict] = []
    seen_keys: set[tuple[int, int, int, str]] = set()

    for comp_a in by_color[endpoint_a]:
        for comp_b in by_color[endpoint_b]:
            if comp_a.square_size != comp_b.square_size:
                continue
            s = comp_a.square_size
            assert s is not None
            for transform in unique_transforms(patch):
                pos_a = positions_of(transform.patch, endpoint_a)
                pos_b = positions_of(transform.patch, endpoint_b)
                if len(pos_a) != 1 or len(pos_b) != 1:
                    continue
                ra, ca = pos_a[0]
                rb, cb = pos_b[0]
                top = comp_a.bbox[0] - ra * s
                left = comp_a.bbox[1] - ca * s
                expected_a = (top + ra * s, left + ca * s, top + (ra + 1) * s - 1, left + (ca + 1) * s - 1)
                expected_b = (top + rb * s, left + cb * s, top + (rb + 1) * s - 1, left + (cb + 1) * s - 1)
                if comp_a.bbox != expected_a or comp_b.bbox != expected_b:
                    continue
                if top < 0 or left < 0 or top + 3 * s > h or left + 3 * s > w:
                    continue
                if not placement_is_clean(grid, transform.patch, bg, top, left, s, comp_a, comp_b):
                    continue
                key = (top, left, s, transform.name)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                overlay_cells = sum(1 for row in transform.patch for value in row if value != bg) * s * s
                placements.append(
                    {
                        "top": top,
                        "left": left,
                        "scale": s,
                        "transform": transform,
                        "overlay_cells": overlay_cells,
                        "endpoint_components": (comp_a.bbox, comp_b.bbox),
                    }
                )
    return placements


def placement_is_clean(
    grid: Grid,
    patch: tuple[tuple[int, ...], ...],
    bg: int,
    top: int,
    left: int,
    scale: int,
    comp_a: Component,
    comp_b: Component,
) -> bool:
    allowed_existing = set(comp_a.cells) | set(comp_b.cells)
    for rr in range(3):
        for cc in range(3):
            value = patch[rr][cc]
            if value == bg:
                continue
            for dr in range(scale):
                for dc in range(scale):
                    r = top + rr * scale + dr
                    c = left + cc * scale + dc
                    current = grid[r][c]
                    if current == bg or current == value or (r, c) in allowed_existing:
                        continue
                    return False
    return True


def apply_placements(grid: Grid, candidate: SourceCandidate) -> Grid:
    out = clone_grid(grid)
    source_bbox = (candidate.top, candidate.left, candidate.top + 2, candidate.left + 2)
    placements = find_placements(
        grid,
        source_bbox,
        candidate.patch,
        candidate.bg,
        candidate.endpoint_a,
        candidate.endpoint_b,
    )
    for placement in placements:
        top = placement["top"]
        left = placement["left"]
        scale = placement["scale"]
        patch = placement["transform"].patch
        for rr in range(3):
            for cc in range(3):
                value = patch[rr][cc]
                if value == candidate.bg:
                    continue
                for dr in range(scale):
                    for dc in range(scale):
                        out[top + rr * scale + dr][left + cc * scale + dc] = value
    return out


def solve_sparse_mask_overlay(grid: Grid) -> Grid:
    """Strict 3x3 source motif plus square marker pair overlay."""
    candidates = source_candidates(grid, require_target_pair=True)
    if not candidates:
        return clone_grid(grid)
    return apply_placements(grid, candidates[0])


def solve_component_bbox_rule(grid: Grid) -> Grid:
    """Connected-component source version. This intentionally tests a narrower hypothesis."""
    bg = dominant_color(grid)
    for comp in components(grid):
        if comp.color == bg:
            continue
        r0, c0, r1, c1 = comp.bbox
        if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
            continue
        patch = patch_at(grid, r0, c0)
        counts = Counter(v for row in patch for v in row if v != bg)
        if len(counts) != 3:
            continue
        candidates = [
            item for item in source_candidates(grid, require_target_pair=True) if item.top == r0 and item.left == c0
        ]
        if candidates:
            return apply_placements(grid, candidates[0])
    return clone_grid(grid)


def solve_marker_motif_copy(grid: Grid) -> Grid:
    """Flexible input-only rule used as the public solve() implementation."""
    candidates = source_candidates(grid, require_target_pair=True)
    if not candidates:
        return clone_grid(grid)

    best_grid = clone_grid(grid)
    best_score = (-1, -1, -1)
    for candidate in candidates[:8]:
        result = apply_placements(grid, candidate)
        changed = count_mismatch(grid, result)
        score = (candidate.score[0], changed, candidate.score[2])
        if score > best_score:
            best_score = score
            best_grid = result
    return best_grid


def solve(grid: Grid) -> Grid:
    return solve_marker_motif_copy(grid)


def count_mismatch(a: Grid, b: Grid) -> int:
    return sum(1 for r in range(len(a)) for c in range(len(a[0])) if a[r][c] != b[r][c])


def iter_examples(task: dict) -> Iterable[tuple[str, int, dict]]:
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(task.get(split, [])):
            yield split, idx, example


def write_grid(path: Path, title: str, grid: Grid) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(title + "\n")
        for row in grid:
            f.write(" ".join(str(v) for v in row) + "\n")


def validate_solver(name: str, solver: Callable[[Grid], Grid], task: dict) -> list[dict]:
    rows: list[dict] = []
    for split, idx, example in iter_examples(task):
        inp = example["input"]
        expected = example.get("output")
        predicted = solver(inp)
        shp = f"{len(inp)}x{len(inp[0])}"
        if expected is None:
            mismatch = ""
            passed = ""
            expected_changed = ""
        else:
            mismatch = count_mismatch(expected, predicted)
            passed = mismatch == 0
            expected_changed = count_mismatch(inp, expected)
            if mismatch:
                fail_base = FAILED_DIR / name / f"{split}_{idx:03d}"
                write_grid(fail_base / "input.txt", "input", inp)
                write_grid(fail_base / "expected.txt", "expected", expected)
                write_grid(fail_base / "predicted.txt", "predicted", predicted)
        rows.append(
            {
                "split": split,
                "index": idx,
                "shape": shp,
                "passed": passed,
                "changed_cells_expected": expected_changed,
                "changed_cells_predicted": count_mismatch(inp, predicted),
                "mismatch_count": mismatch,
                "notes": "",
            }
        )
    return rows


def write_validation_csv(rows: list[dict]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "rule_validation.csv"
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
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for split in ("train", "test", "arc-gen"):
        split_rows = [row for row in rows if row["split"] == split]
        out[f"{split}_total"] = len(split_rows)
        out[f"{split}_passed"] = sum(1 for row in split_rows if row["passed"] is True)
    out["total"] = len(rows)
    out["passed"] = sum(1 for row in rows if row["passed"] is True)
    return out


def validate_all() -> dict:
    task = load_task()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)

    hypotheses: list[tuple[str, Callable[[Grid], Grid], str]] = [
        (
            "h1_sparse_mask_overlay",
            solve_sparse_mask_overlay,
            "Strict 3x3 motif source, opposite-corner endpoint colors, square marker pair overlay.",
        ),
        (
            "h2_connected_component_bbox",
            solve_component_bbox_rule,
            "Connected-component source bbox rule; expected to fail when motif contains background holes.",
        ),
        (
            "h3_marker_driven_motif_copy",
            solve_marker_motif_copy,
            "Input-only source window selection plus D4 transform and square marker-pair scaled overlay.",
        ),
    ]

    hypothesis_summaries: list[tuple[str, dict, str]] = []
    final_rows: list[dict] = []
    for name, solver_fn, description in hypotheses:
        rows = validate_solver(name, solver_fn, task)
        summary = summarize(rows)
        hypothesis_summaries.append((name, summary, description))
        if name == "h3_marker_driven_motif_copy":
            final_rows = rows

    write_validation_csv(final_rows)

    with (REPORT_DIR / "rule_hypotheses.md").open("w", encoding="utf-8") as f:
        f.write("# task158 Rule Hypotheses\n\n")
        for name, summary, description in hypothesis_summaries:
            f.write(f"## {name}\n\n")
            f.write(f"- Description: {description}\n")
            f.write(f"- Passed: {summary['passed']} / {summary['total']}\n")
            f.write(
                f"- Split pass counts: train {summary['train_passed']}/{summary['train_total']}, "
                f"test {summary['test_passed']}/{summary['test_total']}, "
                f"arc-gen {summary['arc-gen_passed']}/{summary['arc-gen_total']}\n\n"
            )

    final_summary = summarize(final_rows)
    with (REPORT_DIR / "rule_validation.md").open("w", encoding="utf-8") as f:
        f.write("# task158 Rule Validation\n\n")
        f.write("Solver: `h3_marker_driven_motif_copy` / `solve()`\n\n")
        f.write("| split | passed | total |\n")
        f.write("| --- | ---: | ---: |\n")
        for split in ("train", "test", "arc-gen"):
            f.write(f"| {split} | {final_summary[f'{split}_passed']} | {final_summary[f'{split}_total']} |\n")
        f.write(f"| all | {final_summary['passed']} | {final_summary['total']} |\n\n")
        f.write("The rule is input-only: detect the dominant background, find a 3x3 motif with two opposite-corner endpoint colors and one fill color, then use square endpoint-color marker components to place scaled D4-transformed sparse overlays.\n")

    return final_summary


if __name__ == "__main__":
    summary = validate_all()
    print(json.dumps(summary, indent=2, sort_keys=True))
