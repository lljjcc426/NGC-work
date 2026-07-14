from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve()
PROJECT = HERE.parent.parent
REPO = HERE.parents[3]
DEFAULT_BASELINE = PROJECT / "config" / "baseline_manifest.json"
DEFAULT_REGISTRY = PROJECT / "config" / "candidate_registry.json"
DEFAULT_OUTPUT = REPO / "workplace C" / "artifacts" / "full400_registered_rebase"
TASK_DATA = REPO / "neurogolf_400_tasks" / "tasks"
SCORER_SOURCE = HERE.parent / "c_score_common.py"


def _score_job(job: tuple[str, str, int]) -> dict[str, Any]:
    task, raw_path, max_examples = job
    sys.path.insert(0, str(HERE.parent))
    from c_score_common import score_onnx

    return asdict(
        score_onnx(
            task,
            Path(raw_path),
            validate_all=max_examples == 0,
            max_examples=max_examples,
        )
    )


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("cache root must be an object")
        return payload
    except Exception:
        backup = path.with_suffix(path.suffix + ".invalid")
        counter = 1
        while backup.exists():
            backup = path.with_suffix(path.suffix + f".invalid{counter}")
            counter += 1
        path.replace(backup)
        return {}


def _score_many(
    jobs: list[tuple[str, Path]],
    *,
    workers: int,
    max_examples: int,
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    from full400_safety import atomic_write_json, validation_cache_key

    results: dict[tuple[str, str], dict[str, Any]] = {}
    pending: list[tuple[str, str, int]] = []
    keys: dict[tuple[str, str], str] = {}
    for task, path in jobs:
        resolved = str(path.resolve())
        key = validation_cache_key(
            task=task,
            model_path=path,
            task_json=TASK_DATA / f"{task}.json",
            scorer_source=SCORER_SOURCE,
            validation_mode="official_full" if max_examples == 0 else "official_screen",
            max_examples=max_examples,
        )
        keys[(task, resolved)] = key
        cached = cache.get(key)
        if cached and Path(cached.get("path", "")).is_file():
            results[(task, resolved)] = cached
        else:
            pending.append((task, resolved, max_examples))

    if pending:
        with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(_score_job, job): job for job in pending}
            for completed, future in enumerate(as_completed(futures), start=1):
                task, raw_path, _ = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = {
                        "task": task,
                        "path": raw_path,
                        "ok": False,
                        "valid_all_checked": max_examples == 0,
                        "examples_checked": 0,
                        "examples_passed": 0,
                        "examples_failed": 0,
                        "memory": None,
                        "params": None,
                        "cost": None,
                        "points": None,
                        "file_size": Path(raw_path).stat().st_size,
                        "sha256": "",
                        "error": f"worker:{type(exc).__name__}:{exc}",
                    }
                resolved = str(Path(raw_path).resolve())
                key = keys[(task, resolved)]
                cache[key] = row
                results[(task, resolved)] = row
                if completed % 25 == 0 or completed == len(pending):
                    print(json.dumps({"stage": "full" if max_examples == 0 else "screen", "completed": completed, "total": len(pending)}), flush=True)
        atomic_write_json(cache_path, cache)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic full-400 package from parent-bound registered candidates."
    )
    parser.add_argument("--baseline-manifest", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate-registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--parent-dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--screen-examples", type=int, default=3)
    parser.add_argument("--parent-score", type=float, help="Compatibility override; must match manifest.")
    parser.add_argument("--parent-zip-sha256", help="Compatibility assertion; must match manifest.")
    parser.add_argument("--exclude-task", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(HERE.parent))
    from candidate_registry import eligible_candidates
    from full400_safety import (
        TASKS,
        assert_complete_onnx_directory,
        atomic_write_json,
        deterministic_zip,
        load_baseline_manifest,
        model_hashes,
        model_set_sha256,
        sha256_file,
        verify_zip,
    )

    baseline = load_baseline_manifest(args.baseline_manifest, args.parent_dir)
    if args.parent_score is not None and abs(args.parent_score - float(baseline["public_score"])) > 1e-9:
        raise RuntimeError("--parent-score does not match baseline manifest")
    if args.parent_zip_sha256 and args.parent_zip_sha256.lower() != baseline["package_sha256"].lower():
        raise RuntimeError("--parent-zip-sha256 does not match baseline manifest")
    parent_dir = Path(baseline["onnx_dir"])
    parent_hashes = baseline["models"]
    eligible = eligible_candidates(args.candidate_registry, parent_hashes)
    excluded = set(args.exclude_task)
    unknown = excluded - set(TASKS)
    if unknown:
        raise RuntimeError(f"unknown excluded tasks: {sorted(unknown)}")

    output_root = args.output_root.resolve()
    output_onnx = output_root / "onnx"
    output_onnx.mkdir(parents=True, exist_ok=True)
    cache_path = output_root / "score_cache.json"
    cache = _load_cache(cache_path)

    screen_jobs: list[tuple[str, Path]] = []
    sources: dict[str, list[tuple[str, Path, dict[str, Any] | None]]] = {}
    for task in TASKS:
        parent = parent_dir / f"{task}.onnx"
        entries: list[tuple[str, Path, dict[str, Any] | None]] = [("parent", parent, None)]
        if task not in excluded:
            for record in eligible.get(task, []):
                entries.append((record["status"], Path(record["candidate_path"]), record))
        sources[task] = entries
        screen_jobs.extend((task, path) for _, path, _ in entries)

    screen = _score_many(
        screen_jobs,
        workers=args.workers,
        max_examples=max(1, args.screen_examples),
        cache=cache,
        cache_path=cache_path,
    )
    full_jobs: list[tuple[str, Path]] = []
    lower: dict[str, list[tuple[str, Path, dict[str, Any]]]] = {}
    for task in TASKS:
        parent_path = parent_dir / f"{task}.onnx"
        parent = screen[(task, str(parent_path.resolve()))]
        if not parent.get("ok") or parent.get("cost") is None:
            raise RuntimeError(f"parent screen failed for {task}: {parent.get('error')}")
        lower[task] = []
        for label, path, record in sources[task][1:]:
            row = screen[(task, str(path.resolve()))]
            if row.get("ok") and row.get("cost") is not None and row["cost"] < parent["cost"]:
                lower[task].append((label, path, record or {}))
                full_jobs.append((task, path))

    full = _score_many(
        full_jobs,
        workers=args.workers,
        max_examples=0,
        cache=cache,
        cache_path=cache_path,
    )
    total_gain = 0.0
    manifest: dict[str, Any] = {"baseline": baseline, "tasks": {}}
    replacements = 0
    for task in TASKS:
        parent_path = parent_dir / f"{task}.onnx"
        parent = screen[(task, str(parent_path.resolve()))]
        winner_label = "parent"
        winner_path = parent_path
        winner = parent
        winner_record = None
        considered = []
        for label, path, record in lower[task]:
            row = full[(task, str(path.resolve()))]
            considered.append({
                "status": label,
                "path": str(path),
                "sha256": row.get("sha256"),
                "cost": row.get("cost"),
                "examples_checked": row.get("examples_checked"),
                "examples_passed": row.get("examples_passed"),
                "ok": row.get("ok"),
                "error": row.get("error", ""),
            })
            if (
                row.get("ok")
                and row.get("examples_checked") == row.get("examples_passed")
                and row.get("cost") is not None
                and row["cost"] < winner["cost"]
            ):
                winner_label, winner_path, winner, winner_record = label, path, row, record
        shutil.copyfile(winner_path, output_onnx / f"{task}.onnx")
        gain = math.log(parent["cost"] / winner["cost"]) if winner["cost"] < parent["cost"] else 0.0
        if gain:
            replacements += 1
            total_gain += gain
        manifest["tasks"][task] = {
            "parent_sha256": parent_hashes[task],
            "parent_cost": parent["cost"],
            "winner_source": winner_label,
            "winner_path": str(winner_path),
            "winner_sha256": sha256_file(winner_path),
            "winner_cost": winner["cost"],
            "delta_points": gain,
            "validation_evidence": winner_record,
            "considered": considered,
        }

    assert_complete_onnx_directory(output_onnx)
    hashes = model_hashes(output_onnx)
    first_zip = output_root / "submission.zip"
    repeat_zip = output_root / "submission.repeat.zip"
    first_sha = deterministic_zip(output_onnx, first_zip)
    repeat_sha = deterministic_zip(output_onnx, repeat_zip)
    if first_sha != repeat_sha:
        raise RuntimeError(f"deterministic ZIP check failed: {first_sha} != {repeat_sha}")
    repeat_zip.unlink()
    zip_check = verify_zip(first_zip, hashes)
    manifest.update({
        "replacement_count": replacements,
        "predicted_point_gain": total_gain,
        "predicted_score": float(baseline["public_score"]) + total_gain,
        "package_sha256": first_sha,
        "model_set_sha256": model_set_sha256(hashes),
        "root_onnx_count": 400,
        "deterministic_zip": True,
        "zip_check": zip_check,
    })
    atomic_write_json(output_root / "package_manifest.json", manifest)
    print(json.dumps({
        "parent_score": baseline["public_score"],
        "replacements": replacements,
        "predicted_point_gain": total_gain,
        "predicted_score": manifest["predicted_score"],
        "package_sha256": first_sha,
        "submission_zip": str(first_zip),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
