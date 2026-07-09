from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from c_score_common import SCORE_DOCS, ensure_dirs, score_onnx, score_result_row, write_csv, write_md


LEDGER = SCORE_DOCS / "30_SCORE_EXPERIMENT_LEDGER.csv"


def append_csv(path: Path, row: dict) -> None:
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    fieldnames = [
        "task",
        "attempt_id",
        "method",
        "old_cost",
        "new_cost",
        "delta_cost",
        "old_points",
        "new_points",
        "delta_points",
        "local_valid",
        "artifact_path",
        "accepted",
        "notes",
    ]
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--old-artifact", required=True)
    parser.add_argument("--new-artifact", required=True)
    parser.add_argument("--method", default="artifact_replacement")
    parser.add_argument("--attempt-id", default="")
    args = parser.parse_args()

    ensure_dirs()
    old = score_onnx(args.task, Path(args.old_artifact), validate_all=True)
    new = score_onnx(args.task, Path(args.new_artifact), validate_all=True)
    accepted = bool(new.ok and old.cost is not None and new.cost is not None and new.cost < old.cost)
    row = {
        "task": args.task,
        "attempt_id": args.attempt_id or f"{args.task}_{datetime.now().strftime('%Y%m%dT%H%M%S')}",
        "method": args.method,
        "old_cost": old.cost,
        "new_cost": new.cost,
        "delta_cost": (old.cost - new.cost) if old.cost is not None and new.cost is not None else "",
        "old_points": old.points,
        "new_points": new.points,
        "delta_points": (new.points - old.points) if old.points is not None and new.points is not None else "",
        "local_valid": new.ok,
        "artifact_path": str(Path(args.new_artifact)),
        "accepted": accepted,
        "notes": new.error,
    }
    append_csv(LEDGER, row)
    md = [
        "# Score Experiment Ledger",
        "",
        "Machine-readable ledger: `30_SCORE_EXPERIMENT_LEDGER.csv`.",
        "",
        "| task | attempt | method | old_cost | new_cost | delta_cost | local_valid | accepted | artifact |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    import csv

    with LEDGER.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            md.append(
                f"| {r['task']} | {r['attempt_id']} | {r['method']} | {r['old_cost']} | {r['new_cost']} | {r['delta_cost']} | {r['local_valid']} | {r['accepted']} | `{r['artifact_path']}` |"
            )
    write_md(SCORE_DOCS / "30_SCORE_EXPERIMENT_LEDGER.md", "\n".join(md))
    print(row)


if __name__ == "__main__":
    main()
