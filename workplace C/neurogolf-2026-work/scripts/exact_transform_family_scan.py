from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import onnx


HERE = Path(__file__).resolve()

TRANSFORMS = {
    "bool_sum_cast": ("collapse_exact_bool_sum_casts.py", "collapse_exact_bool_sum_casts"),
    "not_boolean_consumer": ("fold_not_boolean_consumer.py", "fold"),
    "not_cast_bool": ("fold_not_cast_bool.py", "fold"),
    "not_where": ("fold_not_where.py", "fold"),
    "shared_cast_mask_mul": ("fold_shared_cast_mask_mul.py", "fold"),
    "minmax_cast": ("narrow_minmax_cast.py", "narrow_minmax_cast"),
    "double_complement": ("fold_double_complement.py", "fold_double_complement"),
    "cast_mask_mul": ("fold_cast_mask_mul.py", "fold"),
    "where_bool_cast": ("fold_where_bool_cast.py", "fold"),
    "gather_index_i32": ("narrow_gather_indices.py", "narrow"),
    "commutative_cse": ("merge_commutative_cse.py", "merge"),
}


def _load_transform(filename: str, function: str):
    path = HERE.parent / filename
    name = f"ngc_transform_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function)


def _score(job: tuple[str, str, str, str]) -> dict:
    task, transform, parent_raw, candidate_raw = job
    sys.path.insert(0, str(HERE.parent))
    from c_score_common import score_onnx

    parent = score_onnx(task, Path(parent_raw), validate_all=True)
    candidate = score_onnx(task, Path(candidate_raw), validate_all=True)
    return {
        "task": task,
        "transform": transform,
        "parent_cost": parent.cost,
        "parent_points": parent.points,
        "candidate_cost": candidate.cost,
        "candidate_points": candidate.points,
        "checked": candidate.examples_checked,
        "passed": candidate.examples_passed,
        "ok": candidate.ok,
        "error": candidate.error,
        "candidate": candidate_raw,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--tasks", default="")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    requested = {item.strip() for item in args.tasks.split(",") if item.strip()}
    tasks = sorted(requested or {f"task{i:03d}" for i in range(1, 401)})
    args.work_dir.mkdir(parents=True, exist_ok=True)
    loaded = {
        name: _load_transform(filename, function)
        for name, (filename, function) in TRANSFORMS.items()
    }

    jobs: list[tuple[str, str, str, str]] = []
    for task in tasks:
        parent_path = args.parent_dir / f"{task}.onnx"
        parent = onnx.load(parent_path)
        for name, transform in loaded.items():
            candidate = copy.deepcopy(parent)
            try:
                changed = int(transform(candidate))
                if changed <= 0:
                    continue
                candidate.producer_name = f"ngc_exact_{name}"
                onnx.checker.check_model(candidate, full_check=True)
                candidate = onnx.shape_inference.infer_shapes(candidate, strict_mode=True)
                onnx.checker.check_model(candidate, full_check=True)
                output = args.work_dir / f"{task}_{name}.onnx"
                onnx.save(candidate, output)
                jobs.append((task, name, str(parent_path), str(output)))
            except Exception as exc:
                print(json.dumps({
                    "task": task,
                    "transform": name,
                    "stage": "build",
                    "error": f"{type(exc).__name__}:{exc}",
                }, separators=(",", ":")), flush=True)

    accepted: list[dict] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(_score, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            valid = bool(
                result["ok"]
                and result["checked"] == result["passed"]
                and result["candidate_cost"] is not None
                and result["parent_cost"] is not None
            )
            improved = valid and result["candidate_cost"] < result["parent_cost"]
            result["improved"] = improved
            result["delta_cost"] = (
                result["parent_cost"] - result["candidate_cost"] if improved else 0
            )
            result["delta_points"] = (
                result["candidate_points"] - result["parent_points"] if improved else 0.0
            )
            if improved:
                accepted.append(result)
                print(json.dumps(result, separators=(",", ":")), flush=True)
            else:
                Path(result["candidate"]).unlink(missing_ok=True)

    summary = {
        "tasks_scanned": len(tasks),
        "candidates_built": len(jobs),
        "improved": len(accepted),
        "delta_points_sum_independent": sum(x["delta_points"] for x in accepted),
    }
    print(json.dumps(summary, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
