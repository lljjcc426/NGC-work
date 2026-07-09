from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from infer import constant_submission, make_submission
from train import PROJECT_ROOT, find_sample_submission, find_table_files, read_table, train_baseline


REPORTS_DIR = PROJECT_ROOT / "reports"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
BASELINE_REPORT = REPORTS_DIR / "baseline_report.md"
EXPERIMENTS_CSV = REPORTS_DIR / "experiments.csv"
SUBMISSION_PATH = SUBMISSIONS_DIR / "submission_baseline.csv"


def append_experiment(row: dict[str, str]) -> None:
    EXPERIMENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    exists = EXPERIMENTS_CSV.exists()
    frame = pd.DataFrame([row])
    frame.to_csv(EXPERIMENTS_CSV, mode="a", index=False, header=not exists)


def write_report(payload: dict) -> None:
    lines = [
        "# Baseline Report",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- status: `{payload['status']}`",
        f"- train_file: `{payload.get('train_file', 'UNKNOWN')}`",
        f"- test_file: `{payload.get('test_file', 'UNKNOWN')}`",
        f"- sample_submission: `{payload.get('sample_submission', 'UNKNOWN')}`",
        f"- target_column: `{payload.get('target_column', 'UNKNOWN')}`",
        f"- task_type: `{payload.get('task_type', 'UNKNOWN')}`",
        f"- cv_metric_name: `{payload.get('cv_metric_name', 'UNKNOWN')}`",
        f"- cv_metric_value: `{payload.get('cv_metric_value', 'UNKNOWN')}`",
        f"- submission_path: `{payload.get('submission_path', 'UNKNOWN')}`",
        "",
        "## Notes",
        "",
    ]
    for note in payload.get("notes", []):
        lines.append(f"- {note}")
    lines.extend(["", "## Raw Payload", "", "```json", json.dumps(payload, indent=2), "```", ""])
    BASELINE_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    sample_path = find_sample_submission()
    if sample_path is None:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "blocked",
            "notes": ["No sample_submission file found under data/raw. Download and extract Kaggle data first."],
        }
        write_report(payload)
        print(f"BLOCKED: {payload['notes'][0]}")
        return 1

    sample = read_table(sample_path)
    train_files = find_table_files("train")
    test_files = find_table_files("test")
    if not train_files or not test_files:
        constant_submission(sample, SUBMISSION_PATH)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "sample_copy_baseline",
            "sample_submission": str(sample_path.relative_to(PROJECT_ROOT)),
            "submission_path": str(SUBMISSION_PATH.relative_to(PROJECT_ROOT)),
            "notes": [
                "Train/test tabular files were not found. Wrote a sample-copy submission for format validation only.",
                "Do not submit this baseline until the competition target and metric are confirmed.",
            ],
        }
        write_report(payload)
        append_experiment(
            {
                "exp_id": "baseline_sample_copy",
                "datetime": payload["generated_at"],
                "code_version": "local",
                "features": "none",
                "model": "sample_copy",
                "cv_metric": "",
                "public_lb": "",
                "notes": "format-only baseline because train/test tables were not found",
                "submission_path": str(SUBMISSION_PATH.relative_to(PROJECT_ROOT)),
            }
        )
        print(f"Wrote {SUBMISSION_PATH}")
        return 0

    train_path = train_files[0]
    test_path = test_files[0]
    train = read_table(train_path)
    test = read_table(test_path)
    result = train_baseline(train, test, sample)
    submission = make_submission(sample, test, result, SUBMISSION_PATH)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ok" if result.model is not None else "format_only",
        "train_file": str(train_path.relative_to(PROJECT_ROOT)),
        "test_file": str(test_path.relative_to(PROJECT_ROOT)),
        "sample_submission": str(sample_path.relative_to(PROJECT_ROOT)),
        "target_column": result.target_column,
        "task_type": result.task_type,
        "feature_count": len(result.feature_columns),
        "cv_metric_name": result.cv_metric_name,
        "cv_metric_value": result.cv_metric_value,
        "submission_path": str(SUBMISSION_PATH.relative_to(PROJECT_ROOT)),
        "submission_rows": len(submission),
        "notes": result.notes,
    }
    write_report(payload)
    append_experiment(
        {
            "exp_id": "baseline_auto",
            "datetime": payload["generated_at"],
            "code_version": "local",
            "features": f"{len(result.feature_columns)} auto-detected shared columns",
            "model": f"HistGradientBoosting/{result.task_type}",
            "cv_metric": "" if result.cv_metric_value is None else f"{result.cv_metric_name}={result.cv_metric_value}",
            "public_lb": "",
            "notes": "official metric unknown until competition brief is fetched",
            "submission_path": str(SUBMISSION_PATH.relative_to(PROJECT_ROOT)),
        }
    )
    print(f"Wrote {SUBMISSION_PATH}")
    print(f"Wrote {BASELINE_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
