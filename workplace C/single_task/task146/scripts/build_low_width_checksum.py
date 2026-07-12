from __future__ import annotations

import csv
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


TASK = "task146"
HERE = Path(__file__).resolve()
TASK_DIR = HERE.parents[1]
REPO = HERE.parents[4]
COMMON = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"
DATA = REPO / "neurogolf_400_tasks" / "tasks" / f"{TASK}.json"
BASE = Path(
    r"E:/kagglegolf/submissions/candidates/"
    r"GOLF_20260711_097_v96_plus_task132_task046/onnx/task146.onnx"
)
OUT = TASK_DIR / "onnx" / "task146_candidate.onnx"


def _initializer(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(value, name=name)


def tile_differences() -> tuple[list[tuple[int, int, int]], int]:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    differences: list[tuple[int, int, int]] = []
    examples = 0
    for split in ("train", "test", "arc-gen"):
        for example in data.get(split, []):
            grid = np.asarray(example["input"], dtype=np.int64)
            examples += 1
            for row in (0, 3, 6):
                tile = grid[row : row + 3, :3]
                differences.append(
                    (
                        int(tile[1, 0] - tile[0, 1]),
                        int(tile[2, 0] - tile[0, 2]),
                        int(tile[2, 1] - tile[1, 2]),
                    )
                )
    return differences, examples


def search_coefficients(
    differences: list[tuple[int, int, int]], max_coefficient: int = 32
) -> tuple[tuple[int, int, int], list[dict[str, int]]]:
    unique = sorted(set(differences))
    nonzero = [d for d in unique if d != (0, 0, 0)]
    rows: list[dict[str, int]] = []
    valid: list[tuple[int, int, int]] = []
    # Scale is irrelevant to zero testing, so the first coefficient is fixed to one.
    for b, c in product(range(1, max_coefficient + 1), repeat=2):
        coeff = (1, b, c)
        collisions = sum(
            1 for d in nonzero if sum(x * w for x, w in zip(d, coeff)) == 0
        )
        rows.append(
            {
                "a": coeff[0],
                "b": coeff[1],
                "c": coeff[2],
                "max_abs_coefficient": max(coeff),
                "nonzero_difference_vectors": len(nonzero),
                "zero_collisions": collisions,
            }
        )
        if collisions == 0:
            valid.append(coeff)
    if not valid:
        raise RuntimeError(f"no collision-free coefficients through {max_coefficient}")
    return min(valid, key=lambda x: (max(x), sum(x), x)), rows


def build_onnx(coefficients: tuple[int, int, int], path: Path = OUT) -> Path:
    baseline = onnx.load(BASE)
    graph = baseline.graph
    tail = list(graph.node)[1:]
    del graph.node[:]

    # Crop while converting the one-hot color channels to the scalar color id.
    graph.node.append(
        helper.make_node(
            "Conv",
            ["input", "color_code"],
            ["color_scalar"],
            kernel_shape=[1, 1],
            pads=[0, 0, -21, -27],
        )
    )
    a, b, c = coefficients
    spatial = np.array(
        [[0, -a, -b], [a, 0, -c], [b, c, 0]], dtype=np.float32
    ).reshape(1, 1, 3, 3)
    graph.node.append(
        helper.make_node(
            "Conv",
            ["color_scalar", "spatial_checksum"],
            ["checks"],
            kernel_shape=[3, 3],
            strides=[3, 3],
        )
    )
    graph.node.extend(tail)

    kept = [x for x in graph.initializer if x.name != "w_checksum"]
    del graph.initializer[:]
    graph.initializer.extend(kept)
    graph.initializer.extend(
        [
            _initializer(
                "color_code", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
            ),
            _initializer("spatial_checksum", spatial),
        ]
    )
    graph.name = "task146_low_width_exact_checksum"
    onnx.checker.check_model(baseline)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(baseline, path)
    return path


def main() -> None:
    differences, examples = tile_differences()
    coefficients, search_rows = search_coefficients(differences)
    report_dir = TASK_DIR / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    evidence = report_dir / "checksum_collision_search.csv"
    with evidence.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(search_rows[0]))
        writer.writeheader()
        writer.writerows(search_rows)

    candidate = build_onnx(coefficients)
    sys.path.insert(0, str(COMMON))
    from c_score_common import score_onnx

    old = score_onnx(TASK, BASE, validate_all=True)
    new = score_onnx(TASK, candidate, validate_all=True)
    rows = [
        {
            "model": "baseline",
            "passed": old.examples_passed,
            "checked": old.examples_checked,
            "memory": old.memory,
            "params": old.params,
            "cost": old.cost,
            "points": old.points,
            "ok": old.ok,
            "artifact": str(BASE),
            "notes": "single dense color-position checksum Conv",
        },
        {
            "model": "low_width_exact_checksum",
            "passed": new.examples_passed,
            "checked": new.examples_checked,
            "memory": new.memory,
            "params": new.params,
            "cost": new.cost,
            "points": new.points,
            "ok": new.ok,
            "artifact": str(candidate),
            "notes": f"coefficients={coefficients}; examples={examples}",
        },
    ]
    cost_path = report_dir / "cost_diff_round2.csv"
    with cost_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print({"coefficients": coefficients, "old": old, "new": new})


if __name__ == "__main__":
    main()
