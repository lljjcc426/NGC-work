#!/usr/bin/env python
"""Analyze task233 crop/hole/patch structure for the E-team loop."""
from __future__ import annotations

import csv
import json
import pathlib
from collections import deque

import numpy as np


TASK_JSON = pathlib.Path(r"F:\kaggle\NGC-work\neurogolf_400_tasks\tasks\task233.json")
OUT_CSV = pathlib.Path(__file__).with_name("e_task233_rule_components_20260710.csv")


def components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), max(rs), min(cs), max(cs)


def render(arr: np.ndarray) -> str:
    return "/".join("".join(map(str, row.tolist())) for row in arr)


def analyze_one(split: str, idx: int, ex: dict) -> dict[str, object]:
    inp = np.array(ex["input"], dtype=np.int64)
    out = np.array(ex["output"], dtype=np.int64)
    twos = components(inp == 2)
    main = max(twos, key=len)
    r0, r1, c0, c1 = bbox(main)
    crop = inp[r0 : r1 + 1, c0 : c1 + 1]
    inside_zero = crop == 0
    outside = inp.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = 0
    external_comps = components(outside != 0)
    ext_shapes = []
    for comp in external_comps:
        er0, er1, ec0, ec1 = bbox(comp)
        patch = inp[er0 : er1 + 1, ec0 : ec1 + 1]
        ext_shapes.append(
            {
                "bbox": f"{er0}:{er1},{ec0}:{ec1}",
                "shape": f"{patch.shape[0]}x{patch.shape[1]}",
                "colors": "".join(map(str, sorted(set(patch.ravel()) - {0}))),
                "pattern": render(patch),
            }
        )
    crop_matches_output = crop.shape == out.shape and np.array_equal(crop, out)
    hole_count = int(inside_zero.sum())
    changed_inside = None
    if crop.shape == out.shape:
        changed_inside = int(np.sum(crop != out))
    return {
        "split": split,
        "index": idx,
        "input_shape": f"{inp.shape[0]}x{inp.shape[1]}",
        "output_shape": f"{out.shape[0]}x{out.shape[1]}",
        "main_bbox": f"{r0}:{r1},{c0}:{c1}",
        "main_crop_shape": f"{crop.shape[0]}x{crop.shape[1]}",
        "main_two_cells": len(main),
        "hole_count_in_crop": hole_count,
        "external_component_count": len(external_comps),
        "external_components": json.dumps(ext_shapes, separators=(",", ":")),
        "crop_matches_output": crop_matches_output,
        "changed_inside_crop": "" if changed_inside is None else changed_inside,
    }


def main() -> int:
    data = json.loads(TASK_JSON.read_text(encoding="utf-8"))
    rows = []
    for split, examples in data.items():
        for idx, ex in enumerate(examples):
            rows.append(analyze_one(split, idx, ex))
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {OUT_CSV}")
    for row in rows[:8]:
        print(row["split"], row["index"], row["input_shape"], "->", row["output_shape"], "bbox", row["main_bbox"], "holes", row["hole_count_in_crop"], "external", row["external_component_count"], "crop_equal", row["crop_matches_output"], "changed", row["changed_inside_crop"])
    print(f"crop_matches_output_count={sum(row['crop_matches_output'] for row in rows)}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
