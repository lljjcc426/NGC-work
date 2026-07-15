from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ASSIGNMENTS = REPO / "assignments" / "task_assignment_400.csv"
C_SCRIPTS = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"


def score_job(job: tuple[str, str]) -> dict:
    task, raw_path = job
    sys.path.insert(0, str(C_SCRIPTS))
    from c_score_common import score_onnx

    return asdict(score_onnx(task, Path(raw_path), validate_all=True))


def assigned_tasks() -> list[dict[str, str]]:
    with ASSIGNMENTS.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row["owner"] == "E"]
    return sorted(rows, key=lambda row: int(row["task"].removeprefix("task")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    assignments = assigned_tasks()
    jobs = [
        (row["task"], str(args.onnx_dir / f"{row['task']}.onnx"))
        for row in assignments
    ]
    scored: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(score_job, job): job[0] for job in jobs}
        for future in as_completed(futures):
            task = futures[future]
            scored[task] = future.result()

    rows: list[dict] = []
    for assignment in assignments:
        task = assignment["task"]
        result = scored[task]
        rows.append(
            {
                "task": task,
                "assignment_type": assignment["assignment_type"],
                "ok": result["ok"],
                "examples_checked": result["examples_checked"],
                "examples_passed": result["examples_passed"],
                "memory": result["memory"],
                "params": result["params"],
                "cost": result["cost"],
                "points": result["points"],
                "sha256": result["sha256"],
                "path": result["path"],
                "error": result["error"],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    passed = sum(
        bool(row["ok"] and row["examples_checked"] == row["examples_passed"])
        for row in rows
    )
    print(
        {
            "tasks": len(rows),
            "fully_valid": passed,
            "examples_checked": sum(int(row["examples_checked"]) for row in rows),
            "output": str(args.output),
        }
    )


if __name__ == "__main__":
    main()
