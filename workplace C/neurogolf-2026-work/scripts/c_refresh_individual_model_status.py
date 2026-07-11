from __future__ import annotations

import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parent
WORKPLACE = PROJECT.parent
MANIFEST = WORKPLACE / "task_manifest_C.csv"
LEDGER = WORKPLACE / "score_docs" / "30_SCORE_EXPERIMENT_LEDGER.csv"
SINGLE_TASK = WORKPLACE / "single_task"
OUT_CSV = WORKPLACE / "score_docs" / "C_INDIVIDUAL_MODEL_STATUS.csv"
OUT_MD = WORKPLACE / "score_docs" / "C_INDIVIDUAL_MODEL_STATUS.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def is_dedicated_attempt(row: dict[str, str]) -> bool:
    attempt = row.get("attempt_id", "")
    method = row.get("method", "")
    if attempt == "artifact_scan_top5" or attempt.startswith("surgery_"):
        return False
    if method.startswith("public/local_artifact_reuse") or method.startswith("onnxoptimizer"):
        return False
    return bool(attempt or method)


def numeric(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    manifest = read_csv(MANIFEST)
    ledger = read_csv(LEDGER)
    by_task: dict[str, list[dict[str, str]]] = {}
    for row in ledger:
        if is_dedicated_attempt(row):
            by_task.setdefault(row.get("task", ""), []).append(row)

    rows: list[dict[str, object]] = []
    for item in manifest:
        task = item["task"]
        root = SINGLE_TASK / task
        scripts = sorted((root / "scripts").glob("*.py")) if (root / "scripts").exists() else []
        reports = sorted((root / "reports").glob("*")) if (root / "reports").exists() else []
        attempts = by_task.get(task, [])
        attempted = bool(scripts or reports or attempts)
        accepted_rows = [r for r in attempts if r.get("accepted", "").lower() == "true"]
        cost_rows = [
            (numeric(r.get("new_cost", "")), r)
            for r in attempts
            if r.get("local_valid", "").lower() == "true"
        ]
        cost_rows = [(cost, row) for cost, row in cost_rows if cost is not None]
        best = min(cost_rows, key=lambda pair: pair[0])[1] if cost_rows else {}
        status = "accepted" if accepted_rows else ("attempted_no_gain" if attempted else "unattempted")
        rows.append(
            {
                "task": task,
                "priority_band": item.get("priority_band", ""),
                "baseline_cost": item.get("cost", ""),
                "model_status": status,
                "dedicated_attempt_count": len(attempts),
                "script_count": len(scripts),
                "report_count": len(reports),
                "best_attempt_cost": best.get("new_cost", ""),
                "best_attempt_method": best.get("method", ""),
                "accepted": bool(accepted_rows),
                "task_dir": str(root.relative_to(WORKPLACE)).replace("\\", "/") if root.exists() else "",
            }
        )

    fields = list(rows[0])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    counts = {key: sum(row["model_status"] == key for row in rows) for key in ("accepted", "attempted_no_gain", "unattempted")}
    lines = [
        "# C Individual Model Status",
        "",
        f"- total: {len(rows)}",
        f"- accepted: {counts['accepted']}",
        f"- attempted without accepted gain: {counts['attempted_no_gain']}",
        f"- unattempted: {counts['unattempted']}",
        "",
        "A task counts as attempted only when it has a task-specific script/report or a non-generic experiment ledger entry. Artifact scans and generic optimizer runs do not count.",
        "",
        "| task | priority | baseline cost | status | attempts | scripts | reports | best attempt cost |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['priority_band']} | {row['baseline_cost']} | {row['model_status']} | "
            f"{row['dedicated_attempt_count']} | {row['script_count']} | {row['report_count']} | {row['best_attempt_cost']} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(counts)


if __name__ == "__main__":
    main()
