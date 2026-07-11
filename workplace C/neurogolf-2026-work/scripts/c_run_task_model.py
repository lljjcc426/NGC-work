from __future__ import annotations

import argparse
import csv
from pathlib import Path

from c_task_model_common import (
    TARGET_COST,
    baseline_path,
    candidate_path,
    load_task_model,
    normalize_task,
    score_candidate,
    status_row_from_result,
    task_dir,
    task_model_path,
    upsert_status,
    validate_rule,
    write_csv,
)


def current_cost(task: str) -> int:
    dashboard = task_dir(task).parents[1] / "dashboard" / "task_progress_C.csv"
    with dashboard.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if normalize_task(row["task_id"]) == task:
                return int(float(row["total_cost"]))
    raise KeyError(f"task not found in C dashboard: {task}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--candidate", type=Path, default=None)
    parser.add_argument("--baseline-dir", type=Path, default=None)
    parser.add_argument("--target-cost", type=int, default=TARGET_COST)
    parser.add_argument("--existing-candidate", action="store_true")
    parser.add_argument("--require-target", action="store_true")
    args = parser.parse_args()

    task = normalize_task(args.task)
    model_path = args.model or task_model_path(task)
    output = args.candidate or candidate_path(task)
    module = load_task_model(task, model_path)
    validation = validate_rule(task, module.solve)
    report_dir = task_dir(task) / "reports"
    write_csv(report_dir / "runner_rule_validation.csv", validation.rows)
    if not validation.ok:
        upsert_status(
            {
                "task": task,
                "current_cost": current_cost(task),
                "best_cost": current_cost(task),
                "points": "",
                "local_target_met": "false",
                "rule_valid": "false",
                "onnx_valid": "false",
                "examples_passed": validation.passed,
                "examples_total": validation.total,
                "artifact_path": "",
                "blocker": "python_rule_failed",
            }
        )
        raise SystemExit(f"Python rule failed: {validation.passed}/{validation.total}")

    if not args.existing_candidate:
        output.parent.mkdir(parents=True, exist_ok=True)
        built = Path(module.build_onnx(output))
        if built.resolve() != output.resolve():
            output = built
    if not output.exists():
        raise FileNotFoundError(f"candidate ONNX was not built: {output}")

    result = score_candidate(task, output)
    old_cost = current_cost(task)
    old_points = 25.0 - __import__("math").log(old_cost)
    cost_rows = [
        {
            "task": task,
            "old_artifact": str(baseline_path(task, args.baseline_dir)),
            "new_artifact": str(output),
            "old_cost": old_cost,
            "new_cost": result.cost if result.cost is not None else "",
            "delta_cost": old_cost - result.cost if result.cost is not None else "",
            "old_points": old_points,
            "new_points": result.points if result.points is not None else "",
            "delta_points": result.points - old_points if result.points is not None else "",
            "examples_checked": result.examples_checked,
            "examples_failed": result.examples_failed,
            "local_valid": result.ok,
            "target_cost": args.target_cost,
            "target_met": bool(result.ok and result.cost is not None and result.cost <= args.target_cost),
            "error": result.error,
        }
    ]
    write_csv(report_dir / "runner_cost_diff.csv", cost_rows)
    status = status_row_from_result(
        task,
        current_cost=old_cost,
        rule_validation=validation,
        onnx_result=result,
        artifact=output,
    )
    upsert_status(status)
    print(status)
    if not result.ok:
        raise SystemExit(2)
    if args.require_target and not bool(result.cost is not None and result.cost <= args.target_cost):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
