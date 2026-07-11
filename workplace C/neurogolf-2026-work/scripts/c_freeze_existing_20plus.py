from __future__ import annotations

from c_score_common import CURRENT_BEST_ONNX_DIR, SCORE_DOCS, score_onnx
from c_task_model_common import TARGET_COST, normalize_task, read_status, upsert_status, write_csv


def main() -> None:
    status_rows = read_status()
    tasks = [
        normalize_task(row["task"])
        for row in status_rows
        if int(float(row["best_cost"])) <= TARGET_COST
    ]
    rows = []
    for task in tasks:
        path = CURRENT_BEST_ONNX_DIR / f"{task}.onnx"
        result = score_onnx(task, path, validate_all=True)
        target_met = bool(result.ok and result.cost is not None and result.cost <= TARGET_COST)
        rows.append(
            {
                "task": task,
                "artifact_path": str(path),
                "examples_checked": result.examples_checked,
                "examples_passed": result.examples_passed,
                "examples_failed": result.examples_failed,
                "memory": result.memory,
                "params": result.params,
                "cost": result.cost,
                "points": result.points,
                "target_met": target_met,
                "error": result.error,
            }
        )
        upsert_status(
            {
                "task": task,
                "best_cost": result.cost if result.cost is not None else "",
                "points": result.points if result.points is not None else "",
                "local_target_met": str(target_met).lower(),
                "onnx_valid": str(result.ok).lower(),
                "examples_passed": result.examples_passed,
                "examples_total": result.examples_checked,
                "artifact_path": str(path),
                "blocker": "" if target_met else (result.error or "baseline_regression_failed"),
            }
        )
        print(f"{task}: cost={result.cost} valid={result.ok} target={target_met}", flush=True)

    write_csv(SCORE_DOCS / "C_20PLUS_BASELINE_FREEZE.csv", rows)
    lines = [
        "# C 20+ Baseline Freeze",
        "",
        "| task | passed | examples | memory | params | cost | points |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['target_met']} | {row['examples_passed']}/{row['examples_checked']} | "
            f"{row['memory']} | {row['params']} | {row['cost']} | {row['points']} |"
        )
    (SCORE_DOCS / "C_20PLUS_BASELINE_FREEZE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if len(rows) != 9 or not all(row["target_met"] for row in rows):
        raise SystemExit("20+ baseline freeze failed")


if __name__ == "__main__":
    main()
