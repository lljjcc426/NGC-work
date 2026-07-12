from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


HERE = Path(__file__).resolve()
TASK_DIR = HERE.parents[1]
REPO = HERE.parents[4]
TASK_JSON = REPO / "neurogolf_400_tasks" / "tasks" / "task081.json"
REPORT_DIR = TASK_DIR / "reports"


def examples() -> list[tuple[np.ndarray, np.ndarray]]:
    task = json.loads(TASK_JSON.read_text(encoding="utf-8"))
    return [
        (np.asarray(example["input"]), np.asarray(example["output"]))
        for split in ("train", "test", "arc-gen")
        for example in task[split]
    ]


def unique_windows(kernel: int) -> tuple[np.ndarray, np.ndarray]:
    radius = kernel // 2
    labels: dict[tuple[int, ...], int] = {}
    for input_grid, output_grid in examples():
        cyan = np.pad((input_grid == 8).astype(np.int8), radius)
        for row in range(7):
            for col in range(7):
                window = tuple(cyan[row : row + kernel, col : col + kernel].ravel())
                label = int(output_grid[row, col])
                previous = labels.setdefault(window, label)
                if previous != label:
                    raise AssertionError(f"identical window has labels {previous} and {label}")
    return np.asarray(list(labels), dtype=float), np.asarray(list(labels.values()))


def hard_margin_feasible(x: np.ndarray, y: np.ndarray) -> bool:
    signs = 2 * y.astype(float) - 1
    design = np.column_stack((x, np.ones(len(x))))
    result = linprog(
        np.zeros(design.shape[1]),
        A_ub=-(signs[:, None] * design),
        b_ub=-np.ones(len(x)),
        bounds=[(None, None)] * design.shape[1],
        method="highs",
    )
    return bool(result.success)


def minimal_infeasible_core(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    keep = list(range(len(x)))
    changed = True
    while changed:
        changed = False
        for index in keep.copy():
            trial = [item for item in keep if item != index]
            if trial and not hard_margin_feasible(x[trial], y[trial]):
                keep = trial
                changed = True
    return np.asarray(keep, dtype=int)


def bounded_bucket_ratio(hidden: np.ndarray, labels: np.ndarray, color: int):
    positive = labels == color
    # Scores must put negatives at <= 0 and positives in [1, U].
    # A uint8 output bucket is possible only when U < 3 after rescaling.
    constraints = []
    bounds = []
    for row in hidden[~positive]:
        constraints.append([row[0], row[1], 1, 0])
        bounds.append(0)
    for row in hidden[positive]:
        constraints.append([-row[0], -row[1], -1, 0])
        bounds.append(-1)
        constraints.append([row[0], row[1], 1, -1])
        bounds.append(0)
    return linprog(
        [0, 0, 0, 1],
        A_ub=np.asarray(constraints),
        b_ub=np.asarray(bounds),
        bounds=[(None, None)] * 3 + [(1, None)],
        method="highs",
    )


def two_channel_search(x: np.ndarray, labels: np.ndarray, trials: int) -> dict:
    # Seeded from the best width-2 float ReLU solution found on all 93 windows.
    float_w = np.asarray(
        [
            [-0.023, 0.960, -0.012, -3.044, -3.032, -3.035, -0.018, 0.958, -0.011],
            [-0.011, -3.894, -0.007, 1.968, -5.269, 1.980, 0.002, -3.891, -0.006],
        ]
    )
    float_b = np.asarray([2.087, 3.294])
    rng = np.random.default_rng(81)
    scales = np.linspace(0.5, 20.0, 79)
    checked = 0
    best = None
    for scale in scales:
        base_w = np.rint(float_w * scale).astype(int)
        base_b = np.rint(float_b * scale).astype(int)
        for trial in range(trials):
            if trial == 0:
                weights, bias = base_w.copy(), base_b.copy()
            else:
                weights = base_w + rng.integers(-1, 2, size=base_w.shape)
                bias = base_b + rng.integers(-1, 2, size=base_b.shape)
            if np.max(np.abs(weights)) > 127:
                continue
            checked += 1
            hidden = np.maximum(0, x @ weights.T + bias)
            fits = [bounded_bucket_ratio(hidden, labels, color) for color in (0, 1, 8)]
            if all(result.success for result in fits):
                worst = max(float(result.fun) for result in fits)
                if best is None or worst < best["worst_bucket_ratio"]:
                    best = {
                        "worst_bucket_ratio": worst,
                        "weights": weights.tolist(),
                        "bias": bias.tolist(),
                    }
    return {"checked": checked, "best": best}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--two-channel-trials", type=int, default=0)
    args = parser.parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    cores = []
    for kernel in (3, 5, 7):
        x, labels = unique_windows(kernel)
        for color in (0, 1, 8):
            binary = labels == color
            feasible = hard_margin_feasible(x, binary)
            rows.append(
                {
                    "kernel": kernel,
                    "output_color": color,
                    "unique_windows": len(x),
                    "hard_margin_feasible": feasible,
                    "direct_weight_params": 10 * kernel * kernel,
                }
            )
            if not feasible and kernel in (3, 5):
                for core_id, source_index in enumerate(minimal_infeasible_core(x, binary)):
                    cores.append(
                        {
                            "kernel": kernel,
                            "output_color": color,
                            "core_id": core_id,
                            "label": int(binary[source_index]),
                            "window": "".join(map(str, x[source_index].astype(int))),
                        }
                    )

    with (REPORT_DIR / "hard_margin_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (REPORT_DIR / "hard_margin_counterexamples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cores[0].keys())
        writer.writeheader()
        writer.writerows(cores)

    x3, labels3 = unique_windows(3)
    two_channel = (
        two_channel_search(x3, labels3, args.two_channel_trials)
        if args.two_channel_trials
        else {"checked": 0, "best": None}
    )
    summary = {
        "public_examples": len(examples()),
        "unique_3x3_windows": len(x3),
        "baseline_cost": 464,
        "direct_7x7_weight_lower_bound": 10 * 7 * 7,
        "two_channel_search": two_channel,
    }
    (REPORT_DIR / "hard_margin_search_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
