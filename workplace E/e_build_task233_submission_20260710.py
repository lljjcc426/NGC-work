#!/usr/bin/env python
"""Build the verified task233 submission and update the Kaggle entrypoint zip."""
from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import zipfile


NGC_ROOT = pathlib.Path(r"F:\kaggle\neurogolf-2026")
WORKPLACE = pathlib.Path(__file__).resolve().parent
ENTRYPOINT = NGC_ROOT / "submissions" / "submission.zip"
ARCHIVE = NGC_ROOT / "submissions" / "submission_e_task233_scatter_20260710.zip"
CANDIDATE = (
    WORKPLACE
    / "optimized_onnx"
    / "task233_scatter_remove_20260710"
    / "task233.onnx"
)
MANIFEST = WORKPLACE / "submission_zip_overwrite_e_task233_20260710.json"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    old_sha = sha256(ENTRYPOINT)
    candidate_bytes = CANDIDATE.read_bytes()
    with zipfile.ZipFile(ENTRYPOINT) as source:
        names = source.namelist()
        expected = [f"task{task:03d}.onnx" for task in range(1, 401)]
        if sorted(names) != expected:
            raise RuntimeError("base zip does not contain exactly task001-task400")
        with zipfile.ZipFile(
            ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as target:
            for name in names:
                payload = candidate_bytes if name == "task233.onnx" else source.read(name)
                target.writestr(name, payload)

    with zipfile.ZipFile(ARCHIVE) as check:
        if check.testzip() is not None:
            raise RuntimeError("candidate zip failed CRC validation")
        if sorted(check.namelist()) != expected:
            raise RuntimeError("candidate zip task inventory mismatch")
        embedded_sha = hashlib.sha256(check.read("task233.onnx")).hexdigest()
    candidate_sha = sha256(CANDIDATE)
    if embedded_sha != candidate_sha:
        raise RuntimeError("embedded task233 does not match verified candidate")

    shutil.copy2(ARCHIVE, ENTRYPOINT)
    new_sha = sha256(ENTRYPOINT)
    manifest = {
        "base_entrypoint": str(ENTRYPOINT),
        "base_sha256_before": old_sha,
        "replacement_task": "task233",
        "replacement_onnx": str(CANDIDATE),
        "replacement_sha256": candidate_sha,
        "replacement_cost_before": 31938,
        "replacement_cost_after": 30710,
        "replacement_points_before": "14.628448198",
        "replacement_points_after": "14.667656387",
        "archive": str(ARCHIVE),
        "archive_sha256": sha256(ARCHIVE),
        "entrypoint_sha256_after": new_sha,
        "zip_entries": 400,
        "zip_test": None,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
