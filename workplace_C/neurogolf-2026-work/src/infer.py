from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from train import BaselineResult


def infer_id_columns(sample: pd.DataFrame) -> list[str]:
    columns = [str(c) for c in sample.columns]
    explicit = [c for c in columns if c.lower() in {"id", "row_id", "sample_id", "image_id", "test_id"}]
    if explicit:
        return explicit
    if len(columns) > 1:
        return [columns[0]]
    return []


def constant_submission(sample: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    submission = sample.copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission


def make_submission(sample: pd.DataFrame, test: pd.DataFrame, result: BaselineResult, output_path: Path) -> pd.DataFrame:
    if result.model is None or not result.feature_columns:
        return constant_submission(sample, output_path)

    submission = sample.copy()
    id_columns = infer_id_columns(sample)
    pred_columns = [c for c in sample.columns if c not in id_columns]
    if not pred_columns:
        raise ValueError("No prediction columns found in sample submission.")

    x_test = test[result.feature_columns].copy()
    if result.task_type == "classification":
        if len(pred_columns) == 1:
            pred = result.model.predict(x_test)
            if result.label_encoder is not None:
                pred = result.label_encoder.inverse_transform(pred.astype(int))
            submission[pred_columns[0]] = pred
        else:
            probabilities = result.model.predict_proba(x_test)
            classes = result.label_encoder.classes_.astype(str) if result.label_encoder is not None else np.arange(probabilities.shape[1]).astype(str)
            for column in pred_columns:
                if str(column) in set(classes):
                    idx = int(np.where(classes == str(column))[0][0])
                    submission[column] = probabilities[:, idx]
                else:
                    submission[column] = sample[column].values
    elif result.task_type == "regression" and len(pred_columns) == 1:
        submission[pred_columns[0]] = result.model.predict(x_test)
    else:
        submission = sample.copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission
