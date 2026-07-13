from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from c_score_common import REPO_ROOT, score_onnx


FORBIDDEN_PATH_MARKERS = (
    "cd_archive",
    "neurogolf7300_archive",
    "archive_one_round",
)


def resolve_artifact(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def score_job(task: str, path: str) -> dict:
    return asdict(score_onnx(task, Path(path), validate_all=True))


def extract_base(base_zip: Path, output_root: Path) -> Path:
    base_dir = output_root / "base_7296_04" / "onnx"
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True)
    with zipfile.ZipFile(base_zip) as archive:
        archive.extractall(base_dir)
    files = sorted(base_dir.glob("task*.onnx"))
    if len(files) != 400:
        raise RuntimeError(f"base package must contain 400 ONNX files, found {len(files)}")
    return base_dir


def read_candidates(ledger: Path) -> list[dict[str, str]]:
    with ledger.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    candidates: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if row.get("accepted", "").strip().lower() != "true":
            continue
        if row.get("local_valid", "").strip().lower() != "true":
            continue
        raw_path = row.get("artifact_path", "").strip()
        if not raw_path:
            continue
        artifact = resolve_artifact(raw_path).resolve()
        lowered = str(artifact).lower()
        if any(marker in lowered for marker in FORBIDDEN_PATH_MARKERS):
            continue
        if not artifact.exists():
            continue
        task = row["task"].strip()
        key = (task, str(artifact).lower())
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "task": task,
                "method": row.get("method", ""),
                "artifact_path": str(artifact),
            }
        )
    return candidates


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def greedy_groups(rows: list[dict], count: int) -> list[list[dict]]:
    groups: list[list[dict]] = [[] for _ in range(count)]
    totals = [0.0] * count
    for row in sorted(rows, key=lambda item: -float(item["delta_points"])):
        index = min(range(count), key=lambda item: totals[item])
        groups[index].append(row)
        totals[index] += float(row["delta_points"])
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    base_dir = extract_base(args.base_zip.resolve(), output_root)
    candidates = read_candidates(args.ledger.resolve())

    jobs: dict[tuple[str, str], tuple[str, Path]] = {}
    for row in candidates:
        task = row["task"]
        jobs[(task, str(base_dir / f"{task}.onnx"))] = (task, base_dir / f"{task}.onnx")
        jobs[(task, row["artifact_path"])] = (task, Path(row["artifact_path"]))

    scores: dict[tuple[str, str], dict] = {}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(score_job, task, str(path)): (task, str(path))
            for task, path in jobs.values()
        }
        for future in as_completed(futures):
            key = futures[future]
            scores[key] = future.result()
            result = scores[key]
            print(
                f"{key[0]} {Path(key[1]).name}: ok={result['ok']} "
                f"cost={result['cost']} examples={result['examples_passed']}/{result['examples_checked']}",
                flush=True,
            )

    comparison: list[dict] = []
    for row in candidates:
        task = row["task"]
        base = scores[(task, str(base_dir / f"{task}.onnx"))]
        candidate = scores[(task, row["artifact_path"])]
        old_cost = base.get("cost")
        new_cost = candidate.get("cost")
        delta = ""
        if old_cost is not None and new_cost is not None:
            delta = math.log(old_cost / new_cost)
        comparison.append(
            {
                **row,
                "base_ok": base.get("ok"),
                "base_cost": old_cost,
                "candidate_ok": candidate.get("ok"),
                "candidate_cost": new_cost,
                "examples_checked": candidate.get("examples_checked"),
                "examples_passed": candidate.get("examples_passed"),
                "delta_points": delta,
                "eligible": bool(candidate.get("ok") and old_cost and new_cost and new_cost < old_cost),
                "base_sha256": base.get("sha256"),
                "candidate_sha256": candidate.get("sha256"),
                "error": candidate.get("error", ""),
            }
        )

    comparison_fields = [
        "task",
        "method",
        "artifact_path",
        "base_ok",
        "base_cost",
        "candidate_ok",
        "candidate_cost",
        "examples_checked",
        "examples_passed",
        "delta_points",
        "eligible",
        "base_sha256",
        "candidate_sha256",
        "error",
    ]
    write_csv(output_root / "candidate_comparison.csv", comparison, comparison_fields)

    best_by_task: dict[str, dict] = {}
    for row in comparison:
        if not row["eligible"]:
            continue
        current = best_by_task.get(row["task"])
        if current is None or int(row["candidate_cost"]) < int(current["candidate_cost"]):
            best_by_task[row["task"]] = row
    eligible = sorted(best_by_task.values(), key=lambda row: -float(row["delta_points"]))
    write_csv(output_root / "eligible_best.csv", eligible, comparison_fields)
    if len(eligible) < 5:
        raise RuntimeError(f"need at least five eligible tasks, found {len(eligible)}")

    groups = greedy_groups(eligible, 5)
    cumulative: list[dict] = []
    manifest_groups: list[dict] = []
    for index, group in enumerate(groups, start=1):
        cumulative.extend(group)
        cumulative = sorted(cumulative, key=lambda row: row["task"])
        override_rows = [
            {
                "task_id": row["task"],
                "candidate_model_path": row["artifact_path"],
                "method_family": row["method"],
                "cost_proxy": row["candidate_cost"],
                "risk": "medium",
                "local_valid": "true",
                "candidate_rank": rank,
            }
            for rank, row in enumerate(cumulative, start=1)
        ]
        override_path = output_root / f"batch_{index}_overrides.csv"
        write_csv(
            override_path,
            override_rows,
            [
                "task_id",
                "candidate_model_path",
                "method_family",
                "cost_proxy",
                "risk",
                "local_valid",
                "candidate_rank",
            ],
        )
        manifest_groups.append(
            {
                "batch": index,
                "new_tasks": [row["task"] for row in group],
                "cumulative_tasks": [row["task"] for row in cumulative],
                "batch_expected_delta": sum(float(row["delta_points"]) for row in group),
                "cumulative_expected_delta": sum(float(row["delta_points"]) for row in cumulative),
                "override_csv": str(override_path),
            }
        )

    manifest = {
        "base_zip": str(args.base_zip.resolve()),
        "base_onnx_dir": str(base_dir),
        "eligible_task_count": len(eligible),
        "total_expected_delta": sum(float(row["delta_points"]) for row in eligible),
        "groups": manifest_groups,
    }
    (output_root / "five_batch_plan.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
