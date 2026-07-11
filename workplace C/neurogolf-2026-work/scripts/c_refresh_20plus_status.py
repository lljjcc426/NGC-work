from __future__ import annotations

import csv
from pathlib import Path

from c_score_common import CURRENT_BEST_ONNX_DIR, WORKPLACE_C, points_from_cost
from c_task_model_common import (
    STATUS_FIELDS,
    TARGET_COST,
    candidate_path,
    normalize_task,
    read_status,
    task_model_path,
    write_status,
)


DASHBOARD = WORKPLACE_C / "dashboard" / "task_progress_C.csv"
LEDGER = WORKPLACE_C / "score_docs" / "30_SCORE_EXPERIMENT_LEDGER.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def best_ledger_rows() -> dict[str, dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    for row in read_csv(LEDGER):
        if str(row.get("local_valid", "")).lower() != "true":
            continue
        try:
            cost = int(float(row["new_cost"]))
        except (KeyError, TypeError, ValueError):
            continue
        task = normalize_task(row["task"])
        if task not in best or cost < int(float(best[task]["new_cost"])):
            best[task] = row
    return best


def main() -> None:
    existing = {normalize_task(row["task"]): row for row in read_status()}
    ledger = best_ledger_rows()
    rows = []
    for source in read_csv(DASHBOARD):
        task = normalize_task(source["task_id"])
        current_cost = int(float(source["total_cost"]))
        best_cost = current_cost
        artifact = CURRENT_BEST_ONNX_DIR / f"{task}.onnx"
        rule_valid = ""
        onnx_valid = str(source.get("correctness", "")).lower() == "pass"
        if task in ledger:
            row = ledger[task]
            candidate_cost = int(float(row["new_cost"]))
            candidate_artifact = Path(row.get("artifact_path", ""))
            if not candidate_artifact.is_absolute():
                candidate_artifact = WORKPLACE_C.parent / candidate_artifact
            if candidate_cost < best_cost and candidate_artifact.exists():
                best_cost = candidate_cost
                artifact = candidate_artifact
                rule_valid = "true" if task_model_path(task).exists() else ""
                onnx_valid = str(row.get("local_valid", "")).lower() == "true"
        local_target_met = bool(onnx_valid and best_cost <= TARGET_COST)
        previous = existing.get(task, {})
        blocker = "" if local_target_met else (
            f"cost_above_target:{best_cost}>{TARGET_COST}"
            if task_model_path(task).exists()
            else "task_model_missing"
        )
        rows.append(
            {
                field: value
                for field, value in {
                    "task": task,
                    "current_cost": current_cost,
                    "best_cost": best_cost,
                    "points": points_from_cost(best_cost),
                    "local_target_met": str(local_target_met).lower(),
                    "rule_valid": previous.get("rule_valid", rule_valid),
                    "onnx_valid": str(onnx_valid).lower(),
                    "examples_passed": previous.get("examples_passed", ""),
                    "examples_total": previous.get("examples_total", ""),
                    "public_parent_score": previous.get("public_parent_score", ""),
                    "public_candidate_score": previous.get("public_candidate_score", ""),
                    "public_delta": previous.get("public_delta", ""),
                    "online_verified": previous.get("online_verified", "false"),
                    "artifact_path": str(artifact),
                    "blocker": blocker,
                }.items()
                if field in STATUS_FIELDS
            }
        )
    if len(rows) != 67:
        raise SystemExit(f"expected 67 C tasks, found {len(rows)}")
    write_status(rows)
    print(f"status={WORKPLACE_C / 'score_docs' / 'C_20_POINT_STATUS.csv'}")
    print(f"local_target_met={sum(row['local_target_met'] == 'true' for row in rows)}/67")


if __name__ == "__main__":
    main()
