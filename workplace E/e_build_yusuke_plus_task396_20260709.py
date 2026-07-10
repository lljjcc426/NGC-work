#!/usr/bin/env python
"""Build E-team candidate: yusuketogashi 7267.31 base plus task396 swap."""
from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import zipfile


NGC_ROOT = pathlib.Path(r"F:\kaggle\neurogolf-2026")
WORKPLACE = pathlib.Path(__file__).resolve().parent
BASE_ZIP = (
    NGC_ROOT
    / "external"
    / "source_review_20260709_e"
    / "yusuketogashi_baseline_7267_31"
    / "output"
    / "submission.zip"
)
SOURCE_ZIP = (
    NGC_ROOT
    / "external"
    / "source_review_20260709_e"
    / "franksunp_7266_72"
    / "output"
    / "submission.zip"
)
OUT_ZIP = NGC_ROOT / "submissions" / "submission_yusuke726731_e_task396_20260709.zip"
SUMMARY_JSON = WORKPLACE / "e_yusuke726731_task396_union_summary_20260709.json"


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    with zipfile.ZipFile(BASE_ZIP, "r") as base, zipfile.ZipFile(
        SOURCE_ZIP, "r"
    ) as source, zipfile.ZipFile(
        OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as out:
        base_names = sorted(base.namelist())
        if len(base_names) != 400:
            raise RuntimeError(f"base zip has {len(base_names)} entries, expected 400")
        if "task396.onnx" not in source.namelist():
            raise RuntimeError("source zip missing task396.onnx")
        for name in base_names:
            payload = source.read(name) if name == "task396.onnx" else base.read(name)
            out.writestr(name, payload)

    with zipfile.ZipFile(OUT_ZIP, "r") as out:
        names = sorted(out.namelist())
        bad = out.testzip()

    summary = {
        "base_zip": str(BASE_ZIP),
        "source_zip": str(SOURCE_ZIP),
        "output_zip": str(OUT_ZIP),
        "output_sha256": sha256(OUT_ZIP),
        "entry_count": len(names),
        "missing_tasks": [
            f"task{i:03d}.onnx" for i in range(1, 401) if f"task{i:03d}.onnx" not in names
        ],
        "extra_entries": [
            name
            for name in names
            if not (name.startswith("task") and name.endswith(".onnx") and len(name) == 12)
        ],
        "testzip": bad,
        "swaps": [
            {
                "task": "task396",
                "base_cost": 3566,
                "candidate_cost": 3562,
                "delta_cost": 4,
                "delta_points": 0.0011223345734769907,
                "source": "franksunp/7266-72-lb-compact-onnx-artifact-starter",
                "verified_all_released": True,
            }
        ],
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
