from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


REPO = Path(__file__).resolve().parents[1]
C_SCRIPTS = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"
TASK_DATA = REPO / "neurogolf_400_tasks" / "tasks"
ort.set_default_logger_severity(3)


def session(path: Path, utils) -> ort.InferenceSession:
    model = onnx.load(path)
    sanitized = utils.sanitize_model(model)
    if sanitized is None:
        raise RuntimeError(f"sanitize_model rejected {path}")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    return ort.InferenceSession(
        sanitized.SerializeToString(),
        options,
        providers=["CPUExecutionProvider"],
    )


def run(model: ort.InferenceSession, value: np.ndarray, utils) -> np.ndarray:
    return utils.run_network(model, value)


def permute_colors(value: np.ndarray, permutation: list[int]) -> np.ndarray:
    result = np.empty_like(value)
    for source, target in enumerate(permutation):
        result[:, target] = value[:, source]
    return result


def load_inputs(task: str, utils) -> list[np.ndarray]:
    payload = json.loads((TASK_DATA / f"{task}.json").read_text(encoding="utf-8"))
    values: list[np.ndarray] = []
    for split in ("train", "test", "arc-gen"):
        for example in payload.get(split, []):
            converted = utils.convert_to_numpy(example)
            if converted is not None:
                values.append(converted["input"])
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(C_SCRIPTS))
    from c_score_common import load_official_utils, sha256_file

    utils = load_official_utils()
    rows: list[dict] = []
    for raw in args.candidate:
        task, candidate_raw = raw.split("=", 1)
        parent_path = args.parent_dir / f"{task}.onnx"
        candidate_path = Path(candidate_raw)
        parent_session = session(parent_path, utils)
        candidate_session = session(candidate_path, utils)
        inputs = load_inputs(task, utils)
        rng = random.Random(f"{args.seed}:{task}:{sha256_file(candidate_path)}")
        mismatches = 0
        parent_runtime_errors = 0
        candidate_runtime_errors = 0
        first_mismatch_trial = ""
        first_mismatch_cells = ""
        first_permutation = ""
        for trial in range(args.trials):
            source = inputs[rng.randrange(len(inputs))]
            active = [
                color
                for color in range(1, 10)
                if np.any(source[:, color])
            ]
            targets = active.copy()
            rng.shuffle(targets)
            permutation = list(range(10))
            for source_color, target_color in zip(active, targets):
                permutation[source_color] = target_color
            value = permute_colors(source, permutation)
            try:
                expected = run(parent_session, value, utils)
            except Exception:
                parent_runtime_errors += 1
                continue
            try:
                actual = run(candidate_session, value, utils)
            except Exception:
                candidate_runtime_errors += 1
                mismatches += 1
                if first_mismatch_trial == "":
                    first_mismatch_trial = trial
                    first_mismatch_cells = "runtime_error"
                    first_permutation = ",".join(map(str, permutation))
                continue
            if not np.array_equal(expected, actual):
                mismatches += 1
                if first_mismatch_trial == "":
                    first_mismatch_trial = trial
                    first_mismatch_cells = int(np.count_nonzero(expected != actual))
                    first_permutation = ",".join(map(str, permutation))
        rows.append(
            {
                "task": task,
                "candidate": str(candidate_path),
                "candidate_sha256": sha256_file(candidate_path),
                "trials": args.trials,
                "comparable_trials": args.trials - parent_runtime_errors,
                "mismatches": mismatches,
                "matched": args.trials - parent_runtime_errors - mismatches,
                "parent_runtime_errors": parent_runtime_errors,
                "candidate_runtime_errors": candidate_runtime_errors,
                "first_mismatch_trial": first_mismatch_trial,
                "first_mismatch_cells": first_mismatch_cells,
                "first_permutation": first_permutation,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
