from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import tempfile
import time
import zipfile
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_NG_ROOT = Path(
    os.environ.get(
        "LOCAL_NG_ROOT",
        "/Users/xiaomingjun/Documents/Codex/2026-06-04/https-www-kaggle-com-code-karnakbaevarthur",
    )
)
DOWNLOADS = Path(
    os.environ.get(
        "D_DOWNLOADS",
        "/Users/xiaomingjun/Library/Containers/com.tencent.qq/Data/Downloads",
    )
)
ASSIGNMENTS = REPO_ROOT / "assignments" / "task_assignment_400.csv"
OUT_DIR = REPO_ROOT / "workplace D"
OPT_DIR = OUT_DIR / "optimized_onnx"
SCAN_CSV = OUT_DIR / "d_deep_candidate_scan_20260709.csv"
ACCEPTED_CSV = OUT_DIR / "d_deep_accepted_optimizations_20260709.csv"
REPORT = OUT_DIR / "d_deep_scan_report_20260709.md"

MAX_PER_TASK = int(os.environ.get("D_DEEP_MAX_PER_TASK", "35"))
MAX_BYTES = int(os.environ.get("D_DEEP_MAX_BYTES", "12000"))


def score(cost: int) -> float:
    return max(1.0, 25.0 - math.log(max(1, cost)))


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rel(path: Path) -> str:
    for root in (LOCAL_NG_ROOT, REPO_ROOT, DOWNLOADS):
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
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


def iter_source_roots() -> list[Path]:
    roots = [
        OPT_DIR,
        LOCAL_NG_ROOT / "work",
        LOCAL_NG_ROOT / "outputs",
        LOCAL_NG_ROOT / "submission.zip",
    ]
    if DOWNLOADS.exists():
        roots.append(DOWNLOADS)
    extra = os.environ.get("D_EXTRA_SOURCE_ROOTS", "")
    for value in extra.split(os.pathsep):
        if value:
            roots.append(Path(value).expanduser())
    return [root for root in roots if root.exists()]


def iter_paths(root: Path):
    if root.is_file():
        yield root
    else:
        yield from root.rglob("*")


def collect_candidates(tasks: set[str]) -> tuple[list[dict[str, object]], int]:
    rows: list[dict[str, object]] = []
    raw_count = 0
    for root in iter_source_roots():
        for path in iter_paths(root):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix == ".onnx":
                task = task_name(path.name)
                if task not in tasks:
                    continue
                try:
                    data = path.read_bytes()
                except OSError:
                    continue
                raw_count += 1
                rows.append(
                    {
                        "task": task,
                        "source_kind": "file",
                        "source": rel(path),
                        "sha256": sha_bytes(data),
                        "bytes": len(data),
                        "data": data,
                    }
                )
            elif suffix == ".zip":
                try:
                    with zipfile.ZipFile(path) as archive:
                        for info in archive.infolist():
                            task = task_name(info.filename)
                            if task not in tasks:
                                continue
                            data = archive.read(info)
                            raw_count += 1
                            rows.append(
                                {
                                    "task": task,
                                    "source_kind": "zip",
                                    "source": f"{rel(path)}::{info.filename}",
                                    "source_member": info.filename,
                                    "sha256": sha_bytes(data),
                                    "bytes": len(data),
                                    "data": data,
                                }
                            )
                except Exception as exc:
                    print("zip-error", rel(path), type(exc).__name__, exc, flush=True)
    return rows, raw_count


def source_rank(source: str) -> tuple[int, str]:
    ranked_tokens = (
        "latest_public_20260707",
        "7235_outputs",
        "next3_outputs",
        "latest_",
        "public-kernel-outputs",
        "outputs/neurogolf_v",
        "submission.zip",
    )
    for index, token in enumerate(ranked_tokens):
        if token in source:
            return index, source
    return len(ranked_tokens), source


def select_candidates(
    assignments: dict[str, dict[str, str]],
    unique_by_task: dict[str, dict[str, dict[str, object]]],
) -> dict[str, list[dict[str, object]]]:
    selected: dict[str, list[dict[str, object]]] = {}
    for task, rows_by_hash in unique_by_task.items():
        base_cost = int(float(assignments[task]["cost"]))
        soft_byte_cap = max(MAX_BYTES, min(24000, base_cost * 2))
        rows = list(rows_by_hash.values())
        small_rows = [row for row in rows if int(row["bytes"]) <= soft_byte_cap]
        if len(small_rows) < min(MAX_PER_TASK, 10):
            small_rows = rows
        small_first = sorted(small_rows, key=lambda row: (int(row["bytes"]), source_rank(str(row["source"]))))[
            :MAX_PER_TASK
        ]
        trusted_first = sorted(rows, key=lambda row: (source_rank(str(row["source"])), int(row["bytes"])))[
            : max(5, MAX_PER_TASK // 3)
        ]
        by_hash: dict[str, dict[str, object]] = {}
        for row in small_first + trusted_first:
            by_hash.setdefault(str(row["sha256"]), row)
        selected[task] = sorted(by_hash.values(), key=lambda row: (int(row["bytes"]), source_rank(str(row["source"]))))
    return selected


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    evaluate = bootstrap_validator()
    assignments = load_d_assignments()
    tasks = set(assignments)
    raw, raw_count = collect_candidates(tasks)

    unique_by_task: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in raw:
        unique_by_task[str(row["task"])].setdefault(str(row["sha256"]), row)

    selected = select_candidates(assignments, unique_by_task)
    total_unique = sum(len(rows) for rows in unique_by_task.values())
    total_selected = sum(len(rows) for rows in selected.values())
    print(
        "D deep scan",
        "raw",
        raw_count,
        "unique",
        total_unique,
        "selected",
        total_selected,
        flush=True,
    )

    scan_rows: list[dict[str, object]] = []
    winners: dict[str, dict[str, object]] = {}

    with tempfile.TemporaryDirectory(prefix="ngc_d_deep_") as temp:
        temp_dir = Path(temp)
        ordered_tasks = sorted(assignments, key=lambda task: int(float(assignments[task]["cost"])), reverse=True)
        for task in ordered_tasks:
            base_cost = int(float(assignments[task]["cost"]))
            base_points = float(assignments[task]["points"])
            rows = selected.get(task, [])
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

    previous_rows: list[dict[str, str]] = []
    if ACCEPTED_CSV.exists():
        with ACCEPTED_CSV.open(newline="") as handle:
            previous_rows = list(csv.DictReader(handle))

    accepted = []
    OPT_DIR.mkdir(parents=True, exist_ok=True)
    for task, row in sorted(winners.items(), key=lambda item: int(item[0][4:7])):
        data = row.pop("data")
        out_path = OPT_DIR / f"{task}.onnx"
        out_path.write_bytes(data)  # type: ignore[arg-type]
        accepted.append({**row, "optimized_path": str(out_path.relative_to(REPO_ROOT))})

    fields = [
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
    ]
    write_csv(SCAN_CSV, scan_rows, fields)
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
        "# Workplace D Deep Candidate Scan",
        "",
        f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S %z')}",
        "",
        "## Scope",
        "",
        f"- D primary tasks: {len(assignments)}.",
        f"- Source roots: {', '.join(rel(root) for root in iter_source_roots())}.",
        f"- Raw candidates: {raw_count}; unique candidates: {total_unique}; selected for validation: {total_selected}.",
        f"- Selection budget: `D_DEEP_MAX_PER_TASK={MAX_PER_TASK}`, `D_DEEP_MAX_BYTES={MAX_BYTES}`.",
        "",
        "## Accepted Replacements",
        "",
        f"- Accepted replacements: {len(accepted)}.",
        f"- Aggregate local delta: cost {total_cost}, points {total_points:.9f}.",
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
        lines.append("- None found in the selected deep-scan budget.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Full scan CSV: `{SCAN_CSV.relative_to(REPO_ROOT)}`.",
            f"- Accepted CSV: `{ACCEPTED_CSV.relative_to(REPO_ROOT)}`.",
            f"- Optimized ONNX folder: `{OPT_DIR.relative_to(REPO_ROOT)}`.",
        ]
    )
    if previous_rows and not accepted:
        lines.extend(
            [
                "",
                "## Note",
                "",
                "This deep scan rewrites `optimized_onnx` from the current winner set. If no deep-scan winner is found, restore the prior accepted set before packaging.",
            ]
        )
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("accepted", len(accepted), "cost_delta", total_cost, "points_delta", f"{total_points:.9f}", flush=True)


if __name__ == "__main__":
    main()
