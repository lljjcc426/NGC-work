from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
import zipfile
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_NG_ROOT = Path(os.environ.get("LOCAL_NG_ROOT", "/Users/xiaomingjun/Documents/Codex/2026-06-04/https-www-kaggle-com-code-karnakbaevarthur"))
ASSIGNMENTS = REPO_ROOT / "assignments" / "task_assignment_400.csv"
OUT_DIR = REPO_ROOT / "workplace D"
OPT_DIR = OUT_DIR / "optimized_onnx"
SCAN_CSV = OUT_DIR / "d_candidate_scan_20260709.csv"
ACCEPTED_CSV = OUT_DIR / "d_accepted_optimizations_20260709.csv"
WORKLOG = OUT_DIR / "worklog.md"

SOURCE_ROOTS = [
    LOCAL_NG_ROOT / "work" / "final-v157-stable-plus-task286-on-v154",
    LOCAL_NG_ROOT / "work" / "final-v154-stable-no-task325-on-v151",
    LOCAL_NG_ROOT / "work" / "final-v147-safe-ryosuke-on-v144",
    LOCAL_NG_ROOT / "work" / "public-kernel-outputs",
    LOCAL_NG_ROOT / "work" / "latest_public_20260707",
    LOCAL_NG_ROOT / "work" / "latest_sources_20260707",
    LOCAL_NG_ROOT / "work" / "hand_optimized",
]

MAX_PER_TASK = int(os.environ.get("D_MAX_PER_TASK", "80"))


def score(cost: int) -> float:
    return max(1.0, 25.0 - math.log(max(1, cost)))


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(LOCAL_NG_ROOT))
    except ValueError:
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)


def task_name(name: str) -> str | None:
    stem = Path(name).name
    if not stem.startswith("task") or not stem.endswith(".onnx"):
        return None
    try:
        task_id = int(stem[4:7])
    except ValueError:
        return None
    if not 1 <= task_id <= 400:
        return None
    return f"task{task_id:03d}"


def load_d_assignments() -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with ASSIGNMENTS.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["owner"] == "D" and row["assignment_type"] == "primary":
                rows[row["task"]] = row
    return rows


def bootstrap_validator():
    sys.path.insert(0, str(LOCAL_NG_ROOT / "work"))
    from validate_public_candidates import evaluate  # type: ignore

    return evaluate


def iter_candidates(tasks: set[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for root in SOURCE_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.onnx")):
            task = task_name(path.name)
            if task not in tasks:
                continue
            try:
                data = path.read_bytes()
            except OSError:
                continue
            rows.append(
                {
                    "task": task,
                    "source_kind": "file",
                    "source_path": rel(path),
                    "source": rel(path),
                    "sha256": sha_bytes(data),
                    "bytes": len(data),
                    "data": data,
                }
            )
        for path in sorted(root.rglob("*.zip")):
            try:
                with zipfile.ZipFile(path) as archive:
                    for info in archive.infolist():
                        task = task_name(info.filename)
                        if task not in tasks:
                            continue
                        data = archive.read(info)
                        rows.append(
                            {
                                "task": task,
                                "source_kind": "zip",
                                "source_path": rel(path),
                                "source_member": info.filename,
                                "source": f"{rel(path)}::{info.filename}",
                                "sha256": sha_bytes(data),
                                "bytes": len(data),
                                "data": data,
                            }
                        )
            except Exception as exc:
                print("zip-error", rel(path), type(exc).__name__, exc, flush=True)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    evaluate = bootstrap_validator()
    assignments = load_d_assignments()
    task_set = set(assignments)
    raw = iter_candidates(task_set)
    unique_by_task: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in raw:
        unique_by_task[str(row["task"])].setdefault(str(row["sha256"]), row)

    print("D tasks", len(assignments), "raw", len(raw), "unique", sum(len(v) for v in unique_by_task.values()), flush=True)
    scan_rows: list[dict[str, object]] = []
    winners: dict[str, dict[str, object]] = {}

    with tempfile.TemporaryDirectory(prefix="ngc_d_scan_") as temp:
        temp_dir = Path(temp)
        ordered_tasks = sorted(
            assignments,
            key=lambda task: int(float(assignments[task]["cost"])),
            reverse=True,
        )
        for task in ordered_tasks:
            base_cost = int(float(assignments[task]["cost"]))
            base_points = float(assignments[task]["points"])
            rows = sorted(unique_by_task.get(task, {}).values(), key=lambda row: int(row["bytes"]))[:MAX_PER_TASK]
            for row in rows:
                data = row["data"]  # type: ignore[assignment]
                candidate_path = temp_dir / f"{task}.{row['sha256']}.onnx"
                candidate_path.write_bytes(data)  # type: ignore[arg-type]
                result = evaluate(candidate_path, int(task[4:7]))
                candidate_path.unlink(missing_ok=True)
                candidate_cost = result.get("cost")
                candidate_points = result.get("score")
                delta_cost = ""
                delta_points = ""
                if result.get("valid") and candidate_cost is not None:
                    delta_cost = int(candidate_cost) - base_cost
                    delta_points = float(candidate_points) - base_points  # type: ignore[arg-type]
                out = {
                    "task": task,
                    "priority": assignments[task]["priority_band"],
                    "base_cost": base_cost,
                    "base_points": base_points,
                    "source": row["source"],
                    "source_kind": row["source_kind"],
                    "source_member": row.get("source_member", ""),
                    "sha256": row["sha256"],
                    "bytes": row["bytes"],
                    "valid": result.get("valid", False),
                    "candidate_cost": candidate_cost if candidate_cost is not None else "",
                    "candidate_points": candidate_points if candidate_points is not None else "",
                    "delta_cost": delta_cost,
                    "delta_points": delta_points,
                    "reason": result.get("reason", ""),
                }
                scan_rows.append(out)
                if result.get("valid") and candidate_cost is not None and int(candidate_cost) < base_cost:
                    old = winners.get(task)
                    if old is None or int(candidate_cost) < int(old["candidate_cost"]):
                        winners[task] = {**out, "data": data}
                        print("WIN", task, base_cost, "->", candidate_cost, row["source"], flush=True)
            print("checked", task, len(rows), flush=True)

    accepted = []
    if OPT_DIR.exists():
        shutil.rmtree(OPT_DIR)
    OPT_DIR.mkdir(parents=True, exist_ok=True)
    for task, row in sorted(winners.items(), key=lambda item: int(item[0][4:7])):
        data = row.pop("data")
        out_path = OPT_DIR / f"{task}.onnx"
        out_path.write_bytes(data)  # type: ignore[arg-type]
        accepted.append({**row, "optimized_path": str(out_path.relative_to(REPO_ROOT))})

    write_csv(
        SCAN_CSV,
        scan_rows,
        [
            "task",
            "priority",
            "base_cost",
            "base_points",
            "source",
            "source_kind",
            "source_member",
            "sha256",
            "bytes",
            "valid",
            "candidate_cost",
            "candidate_points",
            "delta_cost",
            "delta_points",
            "reason",
        ],
    )
    write_csv(
        ACCEPTED_CSV,
        accepted,
        [
            "task",
            "priority",
            "base_cost",
            "base_points",
            "candidate_cost",
            "candidate_points",
            "delta_cost",
            "delta_points",
            "source",
            "source_kind",
            "source_member",
            "sha256",
            "bytes",
            "optimized_path",
        ],
    )

    total_points = sum(float(row["delta_points"]) for row in accepted)
    total_cost = sum(int(row["delta_cost"]) for row in accepted)
    lines = [
        "# Workplace D Worklog",
        "",
        f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S %z')}",
        "",
        "## 2026-07-09 D task optimization scan",
        "",
        f"- Scope: {len(assignments)} primary D tasks from `assignments/task_assignment_400.csv`.",
        f"- Candidate roots: {', '.join(rel(path) for path in SOURCE_ROOTS if path.exists())}.",
        f"- Raw candidates: {len(raw)}; unique candidates: {sum(len(v) for v in unique_by_task.values())}.",
        f"- Accepted replacements: {len(accepted)}.",
        f"- Aggregate local delta: cost {total_cost}, points {total_points:.9f}.",
        f"- Full scan CSV: `{SCAN_CSV.relative_to(REPO_ROOT)}`.",
        f"- Accepted CSV: `{ACCEPTED_CSV.relative_to(REPO_ROOT)}`.",
        f"- Optimized ONNX folder: `{OPT_DIR.relative_to(REPO_ROOT)}`.",
        "",
        "Accepted replacements:",
        "",
    ]
    if accepted:
        for row in accepted:
            lines.append(
                f"- `{row['task']}`: cost {row['base_cost']} -> {row['candidate_cost']} "
                f"({int(row['delta_cost']):+d}), points {float(row['delta_points']):+.9f}; "
                f"source `{row['source']}`."
            )
    else:
        lines.append("- None found with lower validated local cost than the assignment baseline.")
    lines.append("")
    WORKLOG.write_text("\n".join(lines), encoding="utf-8")
    print("accepted", len(accepted), "cost_delta", total_cost, "points_delta", f"{total_points:.9f}", flush=True)


if __name__ == "__main__":
    main()
