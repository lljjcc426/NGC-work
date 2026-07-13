from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_PARENT = Path(
    r"E:/kagglegolf/submissions/candidates/GOLF_20260713_SUBMISSION8_REBASE/onnx"
)
DEFAULT_ARCHIVE = Path(r"E:/kagglegolf/data/external/archive_1_ab6515e0/onnx")
DEFAULT_OUTPUT = Path(
    r"E:/kagglegolf/submissions/candidates/GOLF_20260713_SUBMISSION8_ARCHIVE_LOCAL_REBASE"
)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_tasks() -> list[str]:
    return [f"task{index:03d}" for index in range(1, 401)]


def assert_complete_directory(path: Path) -> None:
    actual = {item.stem for item in path.glob("task*.onnx") if item.is_file()}
    expected = set(expected_tasks())
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(
            f"invalid ONNX directory {path}: count={len(actual)} "
            f"missing={missing[:10]} extra={extra[:10]}"
        )


def local_candidates(task: str, roots: list[Path]) -> list[Path]:
    candidates: set[Path] = set()
    for root in roots:
        task_dir = root / "single_task" / task / "onnx"
        if task_dir.exists():
            # Only the canonical artifact is eligible. Task directories also
            # contain public-fit, unsupported-op and failed experimental graphs
            # that can pass the known examples but are not valid replacements.
            canonical = task_dir / f"{task}_candidate.onnx"
            if canonical.is_file():
                candidates.add(canonical.resolve())
    return sorted(candidates, key=lambda path: str(path).lower())


def score_job(job: tuple[str, str, int]) -> dict:
    task, raw_path, max_examples = job
    sys.path.insert(0, str(SCRIPT_DIR))
    from c_score_common import score_onnx

    result = score_onnx(
        task,
        Path(raw_path),
        validate_all=max_examples == 0,
        max_examples=max_examples,
    )
    return asdict(result)


def score_many(
    jobs: list[tuple[str, Path]],
    workers: int,
    max_examples: int,
    cache: dict[str, dict],
) -> dict[tuple[str, str], dict]:
    results: dict[tuple[str, str], dict] = {}
    pending: list[tuple[str, str, int]] = []
    for task, path in jobs:
        sha = sha256_file(path)
        key = f"{task}:{sha}:examples={max_examples}"
        if key in cache:
            results[(task, str(path.resolve()))] = cache[key]
        else:
            pending.append((task, str(path.resolve()), max_examples))

    if pending:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(score_job, job): job for job in pending}
            completed = 0
            for future in as_completed(futures):
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
                        "sha256": sha256_file(Path(raw_path)),
                        "error": f"worker_error:{type(exc).__name__}:{exc}",
                    }
                key = f"{task}:{row['sha256']}:examples={max_examples}"
                cache[key] = row
                results[(task, str(Path(raw_path).resolve()))] = row
                completed += 1
                if completed % 25 == 0 or completed == len(pending):
                    print(
                        json.dumps(
                            {
                                "stage": "full" if max_examples == 0 else "screen",
                                "completed": completed,
                                "total": len(pending),
                            }
                        ),
                        flush=True,
                    )
    return results


def package_directory(source_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for task in expected_tasks():
            path = source_dir / f"{task}.onnx"
            archive.write(path, arcname=path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebase on a COMPLETE 400-model parent and retain only fully validated lower-cost models."
    )
    parser.add_argument("--parent-dir", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--screen-examples", type=int, default=3)
    parser.add_argument(
        "--local-root",
        type=Path,
        action="append",
        default=[],
        help="Workspace group root containing single_task/taskXXX/onnx. Repeatable.",
    )
    parser.add_argument(
        "--exclude-task",
        action="append",
        default=[],
        help="Force this task to remain on the parent model. Repeatable.",
    )
    parser.add_argument(
        "--local-task",
        action="append",
        default=[],
        help="When present, consider local candidates only for these tasks. Repeatable.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    parent_dir = args.parent_dir.resolve()
    archive_dir = args.archive_dir.resolve()
    output_root = args.output_root.resolve()
    output_onnx = output_root / "onnx"
    cache_path = output_root / "score_cache.json"
    manifest_path = output_root / "package_manifest.json"
    zip_path = output_root / "submission.zip"
    local_roots = args.local_root or [
        REPO_ROOT / "workplace A",
        REPO_ROOT / "workplace C",
        REPO_ROOT / "workplace D",
    ]
    excluded_tasks = set(args.exclude_task)
    allowed_local_tasks = set(args.local_task)
    unknown_exclusions = excluded_tasks - set(expected_tasks())
    if unknown_exclusions:
        raise ValueError(f"unknown excluded tasks: {sorted(unknown_exclusions)}")

    assert_complete_directory(parent_dir)
    assert_complete_directory(archive_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_onnx.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    sources: dict[str, list[tuple[str, Path]]] = {}
    screen_jobs: list[tuple[str, Path]] = []
    for task in expected_tasks():
        parent = parent_dir / f"{task}.onnx"
        entries: list[tuple[str, Path]] = [("parent", parent)]
        archive = archive_dir / f"{task}.onnx"
        if task not in excluded_tasks and sha256_file(archive) != sha256_file(parent):
            entries.append(("archive", archive))
        if task not in excluded_tasks and (
            not allowed_local_tasks or task in allowed_local_tasks
        ):
            for path in local_candidates(task, local_roots):
                if sha256_file(path) != sha256_file(parent):
                    entries.append(("local", path))
        deduplicated: dict[str, tuple[str, Path]] = {}
        for label, path in entries:
            deduplicated.setdefault(sha256_file(path), (label, path))
        sources[task] = list(deduplicated.values())
        screen_jobs.extend((task, path) for _, path in sources[task])

    screen = score_many(
        screen_jobs,
        max(1, args.workers),
        max(1, args.screen_examples),
        cache,
    )
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")

    full_jobs: list[tuple[str, Path]] = []
    candidates_by_task: dict[str, list[tuple[str, Path]]] = {}
    for task in expected_tasks():
        entries = sources[task]
        parent_path = next(path for label, path in entries if label == "parent")
        parent = screen[(task, str(parent_path.resolve()))]
        if not parent.get("ok") or parent.get("cost") is None:
            raise RuntimeError(f"parent screen failed for {task}: {parent.get('error', '')}")
        lower: list[tuple[str, Path]] = []
        for label, path in entries:
            if label == "parent":
                continue
            row = screen[(task, str(path.resolve()))]
            if row.get("ok") and row.get("cost") is not None and row["cost"] < parent["cost"]:
                lower.append((label, path))
                full_jobs.append((task, path))
        candidates_by_task[task] = lower

    full = score_many(full_jobs, max(1, args.workers), 0, cache)
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")

    manifest: dict = {
        "parent_dir": str(parent_dir),
        "parent_online_score": 7379.07,
        "parent_zip_sha256": "77b2e974e84ee0212cd22f7114161f7394bbaa70b5f7452cd465e05bbc99de8b",
        "archive_dir": str(archive_dir),
        "archive_zip_sha256": "ab6515e0f82e2eebe82e205db8a62f46042fd4b7941b7d61b72cf47ba9284f87",
        "excluded_tasks": sorted(excluded_tasks),
        "allowed_local_tasks": sorted(allowed_local_tasks),
        "tasks": {},
    }
    total_gain = 0.0
    replacements = 0
    for task in expected_tasks():
        parent_path = parent_dir / f"{task}.onnx"
        parent = screen[(task, str(parent_path.resolve()))]
        winner_label = "parent"
        winner_path = parent_path
        winner = parent
        considered: list[dict] = []
        for label, path in candidates_by_task[task]:
            row = full[(task, str(path.resolve()))]
            considered.append(
                {
                    "source": label,
                    "path": str(path),
                    "sha256": row.get("sha256"),
                    "cost": row.get("cost"),
                    "ok": row.get("ok"),
                    "examples_checked": row.get("examples_checked"),
                    "examples_passed": row.get("examples_passed"),
                    "error": row.get("error", ""),
                }
            )
            if row.get("ok") and row.get("cost") is not None and row["cost"] < winner["cost"]:
                winner_label = label
                winner_path = path
                winner = row
        destination = output_onnx / f"{task}.onnx"
        shutil.copy2(winner_path, destination)
        gain = math.log(parent["cost"] / winner["cost"]) if winner["cost"] < parent["cost"] else 0.0
        if gain > 0:
            replacements += 1
            total_gain += gain
        manifest["tasks"][task] = {
            "parent_sha256": parent["sha256"],
            "parent_cost": parent["cost"],
            "winner_source": winner_label,
            "winner_path": str(winner_path),
            "winner_sha256": winner["sha256"],
            "winner_cost": winner["cost"],
            "predicted_point_gain": gain,
            "fully_validated_replacement": winner_label != "parent",
            "considered_lower_cost_models": considered,
        }

    assert_complete_directory(output_onnx)
    package_directory(output_onnx, zip_path)
    manifest["replacement_count"] = replacements
    manifest["predicted_point_gain"] = total_gain
    manifest["predicted_score"] = 7379.07 + total_gain
    manifest["package_sha256"] = sha256_file(zip_path)
    manifest["root_onnx_count"] = 400
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "parent_score": 7379.07,
                "replacements": replacements,
                "predicted_point_gain": total_gain,
                "predicted_score": manifest["predicted_score"],
                "package_sha256": manifest["package_sha256"],
                "submission_zip": str(zip_path),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
