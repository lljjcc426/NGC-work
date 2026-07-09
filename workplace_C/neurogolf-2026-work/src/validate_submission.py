from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def find_sample_submission() -> Path:
    candidates = sorted(
        p for p in RAW_DIR.rglob("*") if p.is_file() and "sample" in p.name.lower() and "submission" in p.name.lower()
    )
    if not candidates:
        raise FileNotFoundError("No sample_submission file found under data/raw. Download and extract competition data first.")
    return candidates[0]


def read_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t")
    if ext in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if ext in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if ext == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported submission format: {path}")


def infer_id_columns(sample: pd.DataFrame) -> list[str]:
    columns = [str(c) for c in sample.columns]
    explicit = [c for c in columns if c.lower() in {"id", "row_id", "sample_id", "image_id", "test_id"}]
    if explicit:
        return explicit
    if len(columns) > 1:
        return [columns[0]]
    return []


def validate(sample_path: Path, submission_path: Path) -> list[str]:
    errors: list[str] = []
    sample = read_table(sample_path)
    submission = read_table(submission_path)

    sample_columns = [str(c) for c in sample.columns]
    submission_columns = [str(c) for c in submission.columns]
    if submission_columns != sample_columns:
        errors.append(f"Column mismatch. Expected {sample_columns}, got {submission_columns}.")
        return errors

    if len(submission) != len(sample):
        errors.append(f"Row count mismatch. Expected {len(sample)}, got {len(submission)}.")

    id_columns = infer_id_columns(sample)
    for column in id_columns:
        if not submission[column].equals(sample[column]):
            errors.append(f"ID column order/value mismatch in `{column}`.")

    pred_columns = [c for c in sample_columns if c not in id_columns]
    if not pred_columns:
        errors.append("No prediction columns detected after ID column inference.")

    for column in pred_columns:
        series = submission[column]
        if series.isna().any():
            errors.append(f"Prediction column `{column}` contains missing values.")
        if pd.api.types.is_numeric_dtype(series):
            values = series.to_numpy(dtype=float, copy=False)
            if not np.isfinite(values).all():
                errors.append(f"Prediction column `{column}` contains non-finite numeric values.")
        else:
            if series.astype(str).str.len().eq(0).any():
                errors.append(f"Prediction column `{column}` contains empty string labels.")
            sample_non_null = sample[column].dropna()
            if pd.api.types.is_object_dtype(sample[column]) and 1 < sample_non_null.nunique() <= 50:
                allowed = set(sample_non_null.astype(str))
                observed = set(series.astype(str))
                extra = sorted(observed - allowed)
                if extra:
                    errors.append(f"Prediction column `{column}` has labels outside sample set: {extra[:20]}.")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path, help="Submission file to validate.")
    parser.add_argument("--sample", type=Path, default=None, help="Optional explicit sample submission path.")
    args = parser.parse_args()

    try:
        submission_path = args.submission
        if not submission_path.is_absolute():
            submission_path = PROJECT_ROOT / submission_path
        if not submission_path.exists():
            raise FileNotFoundError(f"Submission file does not exist: {submission_path}")

        sample_path = args.sample or find_sample_submission()
        if not sample_path.is_absolute():
            sample_path = PROJECT_ROOT / sample_path

        errors = validate(sample_path, submission_path)
    except Exception as exc:
        print("INVALID SUBMISSION")
        print(f"- {type(exc).__name__}: {exc}")
        return 1

    if errors:
        print("INVALID SUBMISSION")
        for error in errors:
            print(f"- {error}")
        return 1
    print("VALID SUBMISSION")
    print(f"sample={sample_path}")
    print(f"submission={submission_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
