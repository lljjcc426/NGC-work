from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
DEFAULT_PARENT = Path(
    r"E:/kongming/NGC-work/workplace C/artifacts/"
    r"GOLF_20260714_FULL400_ROUND3_REBASE_7379_19/onnx"
)
ASSIGNMENTS = REPO / "assignments" / "task_assignment_400.csv"
TASKS = REPO / "neurogolf_400_tasks" / "tasks"
GENERATORS = REPO / "neurogolf_400_tasks" / "generators"


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
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session.run([output_name], {input_name: array})[0]


def _one_hot(grid: np.ndarray) -> np.ndarray:
    height, width = grid.shape
    if height > 30 or width > 30:
        raise ValueError(f"grid exceeds 30x30: {grid.shape}")
    array = np.zeros((1, 10, 30, 30), dtype=np.float32)
    rows, cols = np.indices((height, width))
    array[0, grid.astype(np.int64), rows, cols] = 1.0
    return array


def _decoded_grid(output: np.ndarray) -> np.ndarray:
    if output.ndim != 4 or output.shape[0] != 1:
        raise ValueError(f"cannot decode output shape {output.shape}")
    if output.shape[1] == 10:
        return np.argmax(output, axis=1)
    return output


def _compare(parent: np.ndarray, candidate: np.ndarray, mode: str) -> tuple[bool, float, int]:
    if parent.shape != candidate.shape:
        return False, float("inf"), int(max(parent.size, candidate.size))
    if mode == "exact":
        equal = np.array_equal(parent, candidate)
    elif mode == "allclose":
        equal = np.allclose(parent, candidate, rtol=0.0, atol=1e-6, equal_nan=False)
    elif mode == "decoded-grid":
        parent = _decoded_grid(parent)
        candidate = _decoded_grid(candidate)
        equal = np.array_equal(parent, candidate)
    elif mode == "sign-mask":
        parent = parent > 0.0
        candidate = candidate > 0.0
        equal = np.array_equal(parent, candidate)
    elif mode == "argmax":
        parent = np.argmax(parent, axis=1)
        candidate = np.argmax(candidate, axis=1)
        equal = np.array_equal(parent, candidate)
    else:
        raise ValueError(mode)
    if parent.dtype == np.bool_ or candidate.dtype == np.bool_:
        maximum = float(np.max(parent != candidate)) if parent.size else 0.0
    else:
        maximum = float(np.max(np.abs(parent.astype(np.float64) - candidate.astype(np.float64)))) if parent.size else 0.0
    different = int(np.count_nonzero(parent != candidate))
    return bool(equal), maximum, different


def _public_inputs(task: str) -> list[np.ndarray]:
    payload = json.loads((TASKS / f"{task}.json").read_text(encoding="utf-8"))
    return [
        np.asarray(example["input"], dtype=np.int64)
        for split in ("train", "test", "arc-gen")
        for example in payload.get(split, [])
        if example.get("input")
        and len(example["input"]) <= 30
        and len(example["input"][0]) <= 30
    ]


def _load_generator(task: str, seed: int) -> Callable[[], np.ndarray]:
    path = GENERATORS / f"{task}.py"
    if not path.is_file():
        raise FileNotFoundError(f"no task generator available: {path}")
    spec = importlib.util.spec_from_file_location(f"ngc_generator_{task}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import generator {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory = getattr(module, "generate", None)
    if factory is None:
        raise RuntimeError(f"generator must export generate(seed=...): {path}")
    rng = np.random.default_rng(seed)

    def next_grid() -> np.ndarray:
        value = factory(seed=int(rng.integers(0, 2**31 - 1)))
        grid = value["input"] if isinstance(value, dict) else value
        return np.asarray(grid, dtype=np.int64)

    return next_grid


def _inputs(task: str, input_mode: str, trials: int, seed: int) -> Iterable[np.ndarray]:
    official = _public_inputs(task)
    if not official:
        raise RuntimeError(f"no official inputs for {task}")
    rng = np.random.default_rng(seed)
    if input_mode == "official":
        yield from official
        return
    if input_mode == "generator":
        generator = _load_generator(task, seed)
        for _ in range(trials):
            yield generator()
        return
    shapes = sorted({grid.shape for grid in official})
    for trial in range(trials):
        if input_mode == "color-permutation":
            yield rng.permutation(10)[official[trial % len(official)]]
        elif input_mode == "random-grid":
            height, width = shapes[trial % len(shapes)]
            yield rng.integers(0, 10, size=(height, width), dtype=np.int64)
        else:
            raise ValueError(input_mode)


def _owners() -> dict[str, str]:
    with ASSIGNMENTS.open(newline="", encoding="utf-8-sig") as handle:
        return {row["task"]: row["owner"] for row in csv.DictReader(handle)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact parent/candidate ONNX equivalence fuzzing.")
    parser.add_argument("--parent-model", type=Path)
    parser.add_argument("--candidate-model", type=Path)
    parser.add_argument("--task")
    parser.add_argument("--parent-dir", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--tasks", default="")
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument(
        "--mode",
        choices=("exact", "allclose", "decoded-grid", "sign-mask", "argmax"),
        default="exact",
    )
    parser.add_argument(
        "--input-mode",
        choices=("official", "color-permutation", "random-grid", "generator"),
        default="color-permutation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(HERE.parent))
    from c_score_common import load_official_utils

    explicit = bool(args.parent_model or args.candidate_model or args.task)
    if explicit and not (args.parent_model and args.candidate_model and args.task):
        raise SystemExit("--parent-model, --candidate-model and --task must be supplied together")
    owners = _owners()
    jobs: list[tuple[str, Path, Path]] = []
    if explicit:
        jobs.append((args.task, args.parent_model, args.candidate_model))
    else:
        tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
        if not tasks:
            raise SystemExit("provide an explicit model pair or --tasks")
        for task in tasks:
            candidate = REPO / f"workplace {owners[task]}" / "single_task" / task / "onnx" / f"{task}_candidate.onnx"
            jobs.append((task, args.parent_dir / f"{task}.onnx", candidate))

    utils = load_official_utils()
    failures = 0
    for task, parent_path, candidate_path in jobs:
        parent_session = _session(parent_path, utils)
        candidate_session = _session(candidate_path, utils)
        passed = 0
        error = ""
        failed_trial = None
        last_detail: dict[str, object] = {}
        inputs = list(_inputs(task, args.input_mode, args.trials, args.seed))
        for trial, grid in enumerate(inputs):
            try:
                array = _one_hot(grid)
                parent_output = _run(parent_session, array)
                candidate_output = _run(candidate_session, array)
                equal, maximum, different = _compare(parent_output, candidate_output, args.mode)
                last_detail = {
                    "input_shape": list(grid.shape),
                    "maximum_absolute_error": maximum,
                    "different_element_count": different,
                    "parent_dtype": str(parent_output.dtype),
                    "candidate_dtype": str(candidate_output.dtype),
                    "parent_output_shape": list(parent_output.shape),
                    "candidate_output_shape": list(candidate_output.shape),
                }
                if not equal:
                    error = "output_mismatch"
                    failed_trial = trial
                    break
            except Exception as exc:
                error = f"runtime:{type(exc).__name__}:{exc}"
                failed_trial = trial
                break
            passed += 1
        if error:
            failures += 1
        print(
            json.dumps(
                {
                    "task": task,
                    "parent_sha": __import__("hashlib").sha256(parent_path.read_bytes()).hexdigest(),
                    "candidate_sha": __import__("hashlib").sha256(candidate_path.read_bytes()).hexdigest(),
                    "mode": args.mode,
                    "input_mode": args.input_mode,
                    "trials": len(inputs),
                    "passed": passed,
                    "failed_trial": failed_trial,
                    "equivalent": not error,
                    **last_detail,
                    "error": error,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
