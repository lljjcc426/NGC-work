#!/usr/bin/env python
"""Compare task007 models embedded in team submission packages."""
from __future__ import annotations

import copy
import csv
import hashlib
import pathlib
import sys
import tempfile
import zipfile


sys.path.extend(
    [
        r"C:\ProgramData\anaconda3\Lib\site-packages",
        r"C:\Users\cc\AppData\Roaming\Python\Python311\site-packages",
    ]
)

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime  # noqa: E402


TASK = 7
TASK_FILE = "task007.onnx"
REPO = pathlib.Path(r"F:\kaggle\NGC-work")
NGC_ROOT = pathlib.Path(r"F:\kaggle\neurogolf-2026")
BASE_ZIP = (
    NGC_ROOT
    / "submissions"
    / "submission_team_high_e_task003_qlinear_20260712.zip"
)
OUT_CSV = pathlib.Path(__file__).with_name("e_task007_team_source_scan_20260712.csv")

sys.path.insert(0, str(NGC_ROOT / "data" / "neurogolf_utils"))
import neurogolf_utils as ng  # noqa: E402


ng._NEUROGOLF_DIR = str((NGC_ROOT / "data").resolve()) + "\\"
onnxruntime.set_default_logger_severity(3)


def fix_names(model: onnx.ModelProto) -> None:
    seen: set[str] = set()
    for node in model.graph.node:
        base = node.name or (node.output[0] if node.output else "node")
        name = base
        suffix = 0
        while name in seen:
            suffix += 1
            name = f"{base}_{suffix}"
        node.name = name
        seen.add(name)


def score(payload: bytes) -> int | None:
    try:
        model = ng.sanitize_model(copy.deepcopy(onnx.load_from_string(payload)))
        if model is None:
            return None
        fix_names(model)
        with tempfile.TemporaryDirectory() as tmp:
            options = onnxruntime.SessionOptions()
            options.enable_profiling = True
            options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
            options.log_severity_level = 3
            options.profile_file_prefix = str(pathlib.Path(tmp) / "task007")
            session = onnxruntime.InferenceSession(model.SerializeToString(), options)
            example = ng.convert_to_numpy(ng.load_examples(TASK)["train"][0])
            ng.run_network(session, example["input"])
            trace_path = session.end_profiling()
            memory, params = ng.score_network(model, trace_path)
        if memory is None or params is None:
            return None
        return int(memory) + int(params)
    except Exception:
        return None


def verify(payload: bytes) -> tuple[int, int]:
    model = ng.sanitize_model(copy.deepcopy(onnx.load_from_string(payload)))
    if model is None:
        return 0, 0
    fix_names(model)
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    session = onnxruntime.InferenceSession(model.SerializeToString(), options)
    examples = ng.load_examples(TASK)
    passed = 0
    total = 0
    for split in ("train", "test", "arc-gen"):
        for example in examples[split]:
            batch = ng.convert_to_numpy(example)
            if batch is None:
                continue
            total += 1
            passed += int(
                np.array_equal(ng.run_network(session, batch["input"]), batch["output"])
            )
    return passed, total


def read_task(path: pathlib.Path) -> bytes | None:
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.read(TASK_FILE)
    except (KeyError, OSError, zipfile.BadZipFile):
        return None


def main() -> int:
    sources = [BASE_ZIP, *sorted(REPO.rglob("submission.zip"))]
    seen: dict[str, tuple[pathlib.Path, bytes]] = {}
    for source in sources:
        payload = read_task(source)
        if payload is None:
            continue
        digest = hashlib.sha256(payload).hexdigest()
        seen.setdefault(digest, (source, payload))

    with zipfile.ZipFile(BASE_ZIP) as archive:
        base_payload = archive.read(TASK_FILE)
    base_sha = hashlib.sha256(base_payload).hexdigest()
    base_cost = score(base_payload)
    rows = []
    for digest, (source, payload) in seen.items():
        candidate_cost = score(payload)
        validation = ""
        accepted = False
        if candidate_cost is not None and base_cost is not None and candidate_cost < base_cost:
            passed, total = verify(payload)
            validation = f"{passed}/{total}"
            accepted = passed == total and total > 0
        rows.append(
            {
                "source": str(source),
                "sha256": digest,
                "is_baseline": digest == base_sha,
                "bytes": len(payload),
                "cost": "" if candidate_cost is None else candidate_cost,
                "delta_cost": ""
                if candidate_cost is None or base_cost is None
                else base_cost - candidate_cost,
                "validation": validation,
                "accepted_lower": accepted,
            }
        )
    rows.sort(key=lambda row: (row["cost"] == "", row["cost"] or 10**9, row["source"]))
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"base_cost={base_cost} unique_models={len(rows)}")
    for row in rows:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
