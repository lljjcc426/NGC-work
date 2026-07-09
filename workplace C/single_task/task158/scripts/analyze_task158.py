from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict, deque
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TASK_DIR = SCRIPT_DIR.parent
REPO_ROOT = TASK_DIR.parents[2]
TASK_JSON = REPO_ROOT / "neurogolf_400_tasks" / "tasks" / "task158.json"
REPORTS = TASK_DIR / "reports"
DEBUG = TASK_DIR / "debug"
CHARS = "0123456789AB"


def shape(grid: list[list[int]]) -> tuple[int, int]:
    return len(grid), len(grid[0]) if grid else 0


def colors(grid: list[list[int]]) -> Counter:
    return Counter(v for row in grid for v in row)


def changed_cells(inp: list[list[int]], out: list[list[int]]) -> list[tuple[int, int, int, int]]:
    h, w = shape(inp)
    return [(r, c, inp[r][c], out[r][c]) for r in range(h) for c in range(w) if inp[r][c] != out[r][c]]


def neighbors(grid: list[list[int]], r: int, c: int, radius: int) -> tuple[tuple[int, ...], ...]:
    h, w = shape(grid)
    bg = colors(grid).most_common(1)[0][0]
    rows = []
    for rr in range(r - radius, r + radius + 1):
        row = []
        for cc in range(c - radius, c + radius + 1):
            row.append(grid[rr][cc] if 0 <= rr < h and 0 <= cc < w else bg)
        rows.append(tuple(row))
    return tuple(rows)


def components(grid: list[list[int]], bg: int) -> list[dict]:
    h, w = shape(grid)
    seen: set[tuple[int, int]] = set()
    comps = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == bg or (r, c) in seen:
                continue
            q = deque([(r, c)])
            seen.add((r, c))
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if (nr, nc) in seen or grid[nr][nc] == bg:
                        continue
                    seen.add((nr, nc))
                    q.append((nr, nc))
            r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
            r1, c1 = max(r for r, _ in cells), max(c for _, c in cells)
            mask = [
                "".join(CHARS[grid[rr][cc]] if (rr, cc) in cells else "." for cc in range(c0, c1 + 1))
                for rr in range(r0, r1 + 1)
            ]
            comps.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "size": len(cells),
                    "shape": (r1 - r0 + 1, c1 - c0 + 1),
                    "colors": dict(colors([[grid[r][c] for r, c in cells]])),
                    "mask": mask,
                    "cells": cells,
                }
            )
    return comps


def grid_text(inp: list[list[int]], out: list[list[int]]) -> str:
    h, w = shape(inp)
    lines = []
    for r in range(h):
        a = "".join(CHARS[v] for v in inp[r])
        b = "".join(CHARS[v] for v in out[r])
        m = "".join("^" if inp[r][c] != out[r][c] else " " for c in range(w))
        lines.append(f"{a}    {b}    {m}")
    return "\n".join(lines)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    DEBUG.mkdir(parents=True, exist_ok=True)
    data = json.loads(TASK_JSON.read_text(encoding="utf-8"))
    rows = []
    changed_color_edges = Counter()
    neighborhood_3 = Counter()
    neighborhood_5 = Counter()
    component_signatures = Counter()
    source_like_count = 0
    total = 0

    for split in ["train", "test", "arc-gen"]:
        for idx, ex in enumerate(data.get(split, [])):
            total += 1
            inp, out = ex["input"], ex["output"]
            h, w = shape(inp)
            bg = colors(inp).most_common(1)[0][0]
            ch = changed_cells(inp, out)
            for r, c, before, after in ch:
                changed_color_edges[(before, after)] += 1
                neighborhood_3[neighbors(inp, r, c, 1)] += 1
                neighborhood_5[neighbors(inp, r, c, 2)] += 1
            comps = components(inp, bg)
            source_like = [c for c in comps if len(c["colors"]) >= 3]
            source_like_count += int(bool(source_like))
            for comp in comps:
                signature = (
                    tuple(sorted(comp["colors"].items())),
                    comp["shape"],
                    tuple(comp["mask"]),
                )
                component_signatures[signature] += 1
            inp_counts = colors(inp)
            out_counts = colors(out)
            rows.append(
                {
                    "split": split,
                    "index": idx,
                    "input_shape": f"{h}x{w}",
                    "output_shape": f"{shape(out)[0]}x{shape(out)[1]}",
                    "background": bg,
                    "input_colors": " ".join(map(str, sorted(inp_counts))),
                    "output_colors": " ".join(map(str, sorted(out_counts))),
                    "output_subset_input": set(out_counts).issubset(set(inp_counts)),
                    "changed_cells": len(ch),
                    "changed_ratio": round(len(ch) / (h * w), 6),
                    "input_counts": dict(sorted(inp_counts.items())),
                    "output_counts": dict(sorted(out_counts.items())),
                    "component_count": len(comps),
                    "source_like_component_count": len(source_like),
                    "component_shapes": ";".join(f"{c['shape'][0]}x{c['shape'][1]}:{c['size']}:{c['colors']}" for c in comps[:12]),
                }
            )
            if split in {"train", "test"} or idx < 20:
                (DEBUG / f"example_{split}_{idx:03d}.txt").write_text(
                    f"{split} {idx}\ninput    output    diff\n{grid_text(inp, out)}\n",
                    encoding="utf-8",
                )

    summary_csv = REPORTS / "task158_examples_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_split = {split: len(data.get(split, [])) for split in ["train", "test", "arc-gen"]}
    changed_counts = [int(r["changed_cells"]) for r in rows]
    ratios = [float(r["changed_ratio"]) for r in rows]
    source_rate = source_like_count / total if total else 0
    md = [
        "# task158 Analysis",
        "",
        f"- task_json: `{TASK_JSON}`",
        f"- examples: train `{by_split['train']}`, test `{by_split['test']}`, arc-gen `{by_split['arc-gen']}`",
        f"- output colors subset of input colors: `{all(str(r['output_subset_input']) == 'True' for r in rows)}`",
        f"- changed cells min/median/max: `{min(changed_counts)}` / `{sorted(changed_counts)[len(changed_counts)//2]}` / `{max(changed_counts)}`",
        f"- changed ratio avg: `{sum(ratios)/len(ratios):.6f}`",
        f"- examples with a 3+ color connected component: `{source_like_count}/{total}` ({source_rate:.3f})",
        "",
        "## Interpretation",
        "",
        "- Background is the dominant color per example.",
        "- Outputs are same-shape and preserve most input cells.",
        "- The recurring structure is a 3x3 motif/template containing two corner marker colors and one bridge/fill color.",
        "- Other same-color rectangular marker components appear elsewhere; the output overlays a rotated/reflected/scaled copy of the motif between matching marker components.",
        "",
        "## Changed Color Edges",
        "",
        "| before | after | count |",
        "| ---: | ---: | ---: |",
    ]
    for (before, after), count in changed_color_edges.most_common():
        md.append(f"| {before} | {after} | {count} |")

    md.extend(["", "## Component Signature Top 20", "", "| count | colors | shape | mask |", "| ---: | --- | --- | --- |"])
    for signature, count in component_signatures.most_common(20):
        color_sig, comp_shape, mask = signature
        md.append(f"| {count} | `{dict(color_sig)}` | `{comp_shape}` | `{';'.join(mask)}` |")

    md.extend(["", "## Neighborhood Features", ""])
    md.append("- 3x3 and 5x5 counters are computed and available for follow-up in this script; the markdown keeps only aggregate interpretations to stay readable.")
    md.append("- High-frequency changed cells are background cells adjacent to existing marker or motif blocks, consistent with mask overlay rather than global recoloring.")
    (REPORTS / "task158_analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(REPORTS / "task158_analysis.md")
    print(summary_csv)


if __name__ == "__main__":
    main()
