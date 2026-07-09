from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


@dataclass
class BaselineResult:
    model: Any | None
    label_encoder: LabelEncoder | None
    task_type: str
    target_column: str | None
    feature_columns: list[str]
    cv_metric_name: str
    cv_metric_value: float | None
    notes: list[str]


def find_table_files(keyword: str) -> list[Path]:
    supported = {".csv", ".tsv", ".parquet", ".pq", ".json", ".jsonl", ".ndjson"}
    return sorted(
        p for p in RAW_DIR.rglob("*") if p.is_file() and keyword in p.name.lower() and p.suffix.lower() in supported
    )


def find_sample_submission() -> Path | None:
    matches = sorted(
        p
        for p in RAW_DIR.rglob("*")
        if p.is_file() and "sample" in p.name.lower() and "submission" in p.name.lower()
    )
    return matches[0] if matches else None


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
    raise ValueError(f"Unsupported table format: {path}")


def infer_id_columns(sample: pd.DataFrame | None, train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    if sample is not None:
        sample_columns = [str(c) for c in sample.columns]
        explicit = [c for c in sample_columns if c.lower() in {"id", "row_id", "sample_id", "image_id", "test_id"}]
        if explicit:
            return [c for c in explicit if c in train.columns or c in test.columns or c in sample.columns]
        if len(sample_columns) > 1:
            return [sample_columns[0]]
    common = [c for c in train.columns if c in test.columns and str(c).lower() in {"id", "row_id", "sample_id"}]
    return [str(c) for c in common]


def infer_target_column(train: pd.DataFrame, test: pd.DataFrame) -> str | None:
    train_cols = {str(c) for c in train.columns}
    test_cols = {str(c) for c in test.columns}
    candidates = sorted(train_cols - test_cols)
    preferred = [c for c in candidates if c.lower() in {"target", "label", "y", "class", "score"}]
    if len(preferred) == 1:
        return preferred[0]
    if len(candidates) == 1:
        return candidates[0]
    return None


def infer_task_type(y: pd.Series) -> str:
    non_null = y.dropna()
    if non_null.empty:
        return "unknown"
    if pd.api.types.is_object_dtype(non_null) or pd.api.types.is_bool_dtype(non_null) or pd.api.types.is_categorical_dtype(non_null):
        return "classification"
    unique_count = non_null.nunique()
    if pd.api.types.is_integer_dtype(non_null) and unique_count <= min(50, max(2, int(len(non_null) * 0.05))):
        return "classification"
    return "regression"


def build_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    categorical_columns = [c for c in frame.columns if c not in numeric_columns]
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, numeric_columns),
            ("cat", categorical, categorical_columns),
        ],
        remainder="drop",
    )


def train_baseline(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame | None = None) -> BaselineResult:
    notes: list[str] = []
    target = infer_target_column(train, test)
    if target is None:
        return BaselineResult(None, None, "unknown", None, [], "not_computed", None, ["Target column was not identified."])

    id_columns = infer_id_columns(sample, train, test)
    feature_columns = [c for c in test.columns if c in train.columns and c not in id_columns]
    if not feature_columns:
        return BaselineResult(None, None, "unknown", target, [], "not_computed", None, ["No shared feature columns found."])

    x = train[feature_columns].copy()
    y = train[target].copy()
    task_type = infer_task_type(y)
    preprocessor = build_preprocessor(x)
    label_encoder: LabelEncoder | None = None

    if task_type == "classification":
        label_encoder = LabelEncoder()
        y_model = label_encoder.fit_transform(y.astype(str))
        model = Pipeline(
            [
                ("preprocess", preprocessor),
                ("model", HistGradientBoostingClassifier(random_state=42, max_iter=100)),
            ]
        )
        metric_name = "accuracy_sanity_check_not_official_metric"
        if len(np.unique(y_model)) > 1 and len(y_model) >= 10:
            min_class = pd.Series(y_model).value_counts().min()
            splits = int(max(2, min(5, min_class)))
            splitter = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
            scores = []
            for train_idx, valid_idx in splitter.split(x, y_model):
                model.fit(x.iloc[train_idx], y_model[train_idx])
                pred = model.predict(x.iloc[valid_idx])
                scores.append(accuracy_score(y_model[valid_idx], pred))
            cv_value = float(np.mean(scores))
        else:
            cv_value = None
            notes.append("Classification CV skipped because the target has one class or too few rows.")
        model.fit(x, y_model)
    elif task_type == "regression":
        y_model = pd.to_numeric(y, errors="coerce")
        keep = y_model.notna()
        x = x.loc[keep].reset_index(drop=True)
        y_model = y_model.loc[keep].to_numpy(dtype=float)
        model = Pipeline(
            [
                ("preprocess", preprocessor),
                ("model", HistGradientBoostingRegressor(random_state=42, max_iter=100)),
            ]
        )
        metric_name = "rmse_sanity_check_not_official_metric"
        if len(y_model) >= 10:
            splits = int(min(5, max(2, len(y_model) // 5)))
            splitter = KFold(n_splits=splits, shuffle=True, random_state=42)
            scores = []
            for train_idx, valid_idx in splitter.split(x):
                model.fit(x.iloc[train_idx], y_model[train_idx])
                pred = model.predict(x.iloc[valid_idx])
                scores.append(mean_squared_error(y_model[valid_idx], pred, squared=False))
            cv_value = float(np.mean(scores))
        else:
            cv_value = None
            notes.append("Regression CV skipped because there are too few labeled rows.")
        model.fit(x, y_model)
    else:
        return BaselineResult(None, None, "unknown", target, feature_columns, "not_computed", None, notes)

    notes.append("Local CV metric is a sanity check only until the official Kaggle metric is fetched.")
    return BaselineResult(model, label_encoder, task_type, target, feature_columns, metric_name, cv_value, notes)
