from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
DEFAULT_PARENT = Path(
    r"E:/kagglegolf/submissions/candidates/"
    r"GOLF_20260713_SUBMISSION8_REBASE/onnx"
)
ASSIGNMENTS = REPO / "assignments" / "task_assignment_400.csv"
TASKS = REPO / "neurogolf_400_tasks" / "tasks"


def _session(path: Path, utils) -> ort.InferenceSession:
    model = utils.sanitize_model(onnx.load(path))
    if model is None:
        raise RuntimeError(f"sanitizer rejected {path}")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def _run(session: ort.InferenceSession, array: np.ndarray) -> np.ndarray:
    return session.run(["output"], {"input": array})[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-dir", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument(
        "--mode",
        choices=("public-input", "color-permutation", "random-grid"),
        default="color-permutation",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(HERE.parent))
    from c_score_common import load_official_utils

    with ASSIGNMENTS.open(newline="", encoding="utf-8-sig") as handle:
        owners = {row["task"]: row["owner"] for row in csv.DictReader(handle)}
    utils = load_official_utils()
    rng = np.random.default_rng(args.seed)

    failures = 0
    for task in [item.strip() for item in args.tasks.split(",") if item.strip()]:
        candidate = (
            REPO
            / f"workplace {owners[task]}"
            / "single_task"
            / task
            / "onnx"
            / f"{task}_candidate.onnx"
        )
        parent_session = _session(args.parent_dir / f"{task}.onnx", utils)
        candidate_session = _session(candidate, utils)
        examples = json.loads((TASKS / f"{task}.json").read_text(encoding="utf-8"))
        inputs = [
            item["input"]
            for split in ("train", "test", "arc-gen")
            for item in examples.get(split, [])
            if item.get("input")
            and len(item["input"]) <= 30
            and len(item["input"][0]) <= 30
        ]
        shapes = {(len(grid), len(grid[0])) for grid in inputs}
        shape_list = sorted(shapes)
        passed = 0
        error = ""
        trial_count = len(inputs) if args.mode == "public-input" else args.trials
        for trial in range(trial_count):
            if args.mode == "public-input":
                grid = np.asarray(inputs[trial], dtype=np.int64)
                height, width = grid.shape
                colors = grid
            elif args.mode == "color-permutation":
                grid = np.asarray(inputs[trial % len(inputs)], dtype=np.int64)
                height, width = grid.shape
                colors = rng.permutation(10)[grid]
            else:
                height, width = shape_list[trial % len(shape_list)]
                colors = rng.integers(0, 10, size=(height, width))
            array = np.zeros((1, 10, 30, 30), dtype=np.float32)
            rows, cols = np.indices((height, width))
            array[0, colors, rows, cols] = 1.0
            try:
                parent_output = _run(parent_session, array)
                candidate_output = _run(candidate_session, array)
            except Exception as exc:
                error = f"runtime:{type(exc).__name__}:{exc}"
                break
            if not np.array_equal(parent_output > 0.0, candidate_output > 0.0):
                mismatch = int(
                    np.count_nonzero((parent_output > 0.0) != (candidate_output > 0.0))
                )
                error = f"output_mismatch:trial={trial}:cells={mismatch}"
                break
            passed += 1
        if error:
            failures += 1
        print(
            json.dumps(
                {
                    "task": task,
                    "trials": trial_count,
                    "mode": args.mode,
                    "passed": passed,
                    "equivalent": not error,
                    "error": error,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
