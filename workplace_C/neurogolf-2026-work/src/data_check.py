from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"
PROFILE_MD = REPORTS_DIR / "data_profile.md"
PROFILE_JSON = REPORTS_DIR / "data_profile.json"

TABLE_EXTENSIONS = {".csv", ".tsv", ".parquet", ".pq", ".json", ".jsonl", ".ndjson"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
ARRAY_EXTENSIONS = {".npy", ".npz"}
TEXT_EXTENSIONS = {".txt", ".md", ".yaml", ".yml", ".json", ".jsonl", ".ndjson"}


@dataclass
class FileProfile:
    path: str
    size_bytes: int
    kind: str
    extension: str
    rows: int | None = None
    columns: int | None = None
    column_names: list[str] = field(default_factory=list)
    missing_rate: dict[str, float] = field(default_factory=dict)
    sample_rows: list[dict[str, Any]] = field(default_factory=list)
    shape: list[int] | None = None
    notes: list[str] = field(default_factory=list)
    error: str | None = None


def rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def safe_json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def dataframe_sample(df: pd.DataFrame, max_rows: int = 5) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in df.head(max_rows).to_dict(orient="records"):
        records.append({str(k): safe_json_value(v) for k, v in row.items()})
    return records


def count_csv_rows(path: Path, delimiter: str = ",") -> int | None:
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            row_count = sum(1 for _ in reader)
        return max(row_count - 1, 0)
    except Exception:
        return None


def profile_table(path: Path, ext: str) -> FileProfile:
    profile = FileProfile(path=rel(path), size_bytes=path.stat().st_size, kind="table", extension=ext)
    try:
        if ext == ".csv":
            df = pd.read_csv(path, nrows=1000)
            profile.rows = count_csv_rows(path, ",")
        elif ext == ".tsv":
            df = pd.read_csv(path, sep="\t", nrows=1000)
            profile.rows = count_csv_rows(path, "\t")
        elif ext in {".parquet", ".pq"}:
            df = pd.read_parquet(path)
            profile.rows = len(df)
        elif ext in {".jsonl", ".ndjson"}:
            df = pd.read_json(path, lines=True, nrows=1000)
            profile.rows = count_csv_rows(path, "\n")
        else:
            df = pd.read_json(path)
            profile.rows = len(df)

        profile.columns = len(df.columns)
        profile.column_names = [str(c) for c in df.columns]
        missing = df.isna().mean(numeric_only=False).to_dict()
        profile.missing_rate = {str(k): round(float(v), 6) for k, v in missing.items()}
        profile.sample_rows = dataframe_sample(df)
        if profile.rows is None:
            profile.rows = len(df)
            profile.notes.append("Row count is based on the loaded sample or parsed object.")
    except Exception as exc:
        profile.error = f"{type(exc).__name__}: {exc}"
    return profile


def profile_array(path: Path, ext: str) -> FileProfile:
    profile = FileProfile(path=rel(path), size_bytes=path.stat().st_size, kind="array", extension=ext)
    try:
        if ext == ".npy":
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            profile.shape = list(arr.shape)
            profile.notes.append(f"dtype={arr.dtype}")
        else:
            data = np.load(path, allow_pickle=False)
            profile.notes.extend([f"{name}: shape={data[name].shape}, dtype={data[name].dtype}" for name in data.files])
    except Exception as exc:
        profile.error = f"{type(exc).__name__}: {exc}"
    return profile


def profile_image(path: Path, ext: str) -> FileProfile:
    profile = FileProfile(path=rel(path), size_bytes=path.stat().st_size, kind="image", extension=ext)
    try:
        from PIL import Image

        with Image.open(path) as image:
            profile.shape = [int(image.height), int(image.width)]
            profile.notes.append(f"mode={image.mode}")
    except Exception as exc:
        profile.error = f"{type(exc).__name__}: {exc}"
    return profile


def profile_text(path: Path, ext: str) -> FileProfile:
    profile = FileProfile(path=rel(path), size_bytes=path.stat().st_size, kind="text", extension=ext)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        profile.rows = len(lines)
        profile.notes.append("preview=" + repr("\n".join(lines[:5]))[:500])
    except Exception as exc:
        profile.error = f"{type(exc).__name__}: {exc}"
    return profile


def profile_file(path: Path) -> FileProfile:
    ext = path.suffix.lower()
    if ext in TABLE_EXTENSIONS:
        return profile_table(path, ext)
    if ext in ARRAY_EXTENSIONS:
        return profile_array(path, ext)
    if ext in IMAGE_EXTENSIONS:
        return profile_image(path, ext)
    if ext in TEXT_EXTENSIONS:
        return profile_text(path, ext)
    return FileProfile(path=rel(path), size_bytes=path.stat().st_size, kind="binary_or_unknown", extension=ext)


def find_named_files(files: list[Path], *needles: str) -> list[str]:
    matches = []
    for path in files:
        lowered = path.name.lower()
        if all(needle in lowered for needle in needles):
            matches.append(rel(path))
    return sorted(matches)


def read_first_table(path: Path) -> pd.DataFrame | None:
    ext = path.suffix.lower()
    try:
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
    except Exception:
        return None
    return None


def infer_roles(files: list[Path]) -> dict[str, Any]:
    sample_paths = [Path(PROJECT_ROOT, p) for p in find_named_files(files, "sample", "submission")]
    train_paths = [Path(PROJECT_ROOT, p) for p in find_named_files(files, "train")]
    test_paths = [Path(PROJECT_ROOT, p) for p in find_named_files(files, "test")]

    roles: dict[str, Any] = {
        "sample_submission_files": [rel(p) for p in sample_paths],
        "train_files": [rel(p) for p in train_paths],
        "test_files": [rel(p) for p in test_paths],
        "id_columns": [],
        "target_columns": [],
        "submission_prediction_columns": [],
        "uncertain": [],
    }

    sample_df = read_first_table(sample_paths[0]) if sample_paths else None
    train_df = read_first_table(train_paths[0]) if train_paths else None
    test_df = read_first_table(test_paths[0]) if test_paths else None

    if sample_df is not None:
        columns = [str(c) for c in sample_df.columns]
        id_candidates = [c for c in columns if c.lower() in {"id", "row_id", "sample_id", "image_id", "test_id"}]
        if not id_candidates and len(columns) > 1:
            id_candidates = [columns[0]]
            roles["uncertain"].append("Using the first sample submission column as ID by convention; verify against competition docs.")
        roles["id_columns"] = id_candidates
        roles["submission_prediction_columns"] = [c for c in columns if c not in id_candidates]
    else:
        roles["uncertain"].append("No sample_submission table was found.")

    if train_df is not None and test_df is not None:
        train_cols = {str(c) for c in train_df.columns}
        test_cols = {str(c) for c in test_df.columns}
        diff = sorted(train_cols - test_cols)
        preferred = [c for c in diff if c.lower() in {"target", "label", "y", "class", "score"}]
        roles["target_columns"] = preferred or diff
        if len(roles["target_columns"]) != 1:
            roles["uncertain"].append("Target column could not be uniquely identified from train/test column differences.")
    else:
        roles["uncertain"].append("Train/test table pair was not found or could not be loaded.")

    return roles


def write_markdown(profiles: list[FileProfile], roles: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Data Profile")
    lines.append("")
    lines.append("Generated by `python src/data_check.py`.")
    lines.append("")
    lines.append("## Detected Roles")
    lines.append("")
    for key, value in roles.items():
        lines.append(f"- `{key}`: `{json.dumps(value, ensure_ascii=False)}`")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    if not profiles:
        lines.append("No files found under `data/raw`.")
    for item in profiles:
        lines.append(f"### `{item.path}`")
        lines.append("")
        lines.append(f"- kind: `{item.kind}`")
        lines.append(f"- extension: `{item.extension or 'UNKNOWN'}`")
        lines.append(f"- size_bytes: `{item.size_bytes}`")
        if item.rows is not None:
            lines.append(f"- rows: `{item.rows}`")
        if item.columns is not None:
            lines.append(f"- columns: `{item.columns}`")
        if item.shape is not None:
            lines.append(f"- shape: `{item.shape}`")
        if item.column_names:
            lines.append(f"- column_names: `{item.column_names}`")
        if item.missing_rate:
            lines.append(f"- missing_rate_sample: `{item.missing_rate}`")
        if item.notes:
            for note in item.notes:
                lines.append(f"- note: {note}")
        if item.error:
            lines.append(f"- error: `{item.error}`")
        if item.sample_rows:
            lines.append("")
            lines.append("Sample rows:")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(item.sample_rows, indent=2, ensure_ascii=False))
            lines.append("```")
        lines.append("")
    PROFILE_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in RAW_DIR.rglob("*") if p.is_file() and p.name not in {".gitkeep"}])
    profiles = [profile_file(path) for path in files]
    roles = infer_roles(files)
    payload = {
        "raw_dir": rel(RAW_DIR),
        "file_count": len(files),
        "roles": roles,
        "files": [asdict(profile) for profile in profiles],
    }
    PROFILE_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(profiles, roles)
    print(f"Wrote {rel(PROFILE_MD)}")
    print(f"Wrote {rel(PROFILE_JSON)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
