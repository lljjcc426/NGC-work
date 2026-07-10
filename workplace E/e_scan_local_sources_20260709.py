#!/usr/bin/env python
"""Scan local zip/model pools for E-task improvements over the 7267.31 base."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import math
import pathlib
import sys
import tempfile
import zipfile

import numpy as np
import onnx
import onnxruntime


REPO = pathlib.Path(__file__).resolve().parents[1]
NGC_ROOT = pathlib.Path(r"F:\kaggle\neurogolf-2026")
ASSIGNMENT = REPO / "assignments" / "task_assignment_400.csv"
BASE_ZIP = (
    NGC_ROOT
    / "external"
    / "source_review_20260709_e"
    / "yusuketogashi_baseline_7267_31"
    / "output"
    / "submission.zip"
)
OUT_CSV = pathlib.Path(__file__).with_name("e_local_source_scan_20260709.csv")

sys.path.insert(0, str(NGC_ROOT / "data" / "neurogolf_utils"))
import neurogolf_utils as ng  # noqa: E402


ng._NEUROGOLF_DIR = str((NGC_ROOT / "data").resolve()) + "\\"
onnxruntime.set_default_logger_severity(3)


def load_e_tasks() -> list[int]:
    with ASSIGNMENT.open(newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row["owner"] == "E"]
    return sorted({int(row["task"].replace("task", "")) for row in rows})


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


def points(cost: int) -> float:
    return max(1.0, 25.0 - math.log(cost)) if cost > 0 else 25.0


def score_cost(model_bytes: bytes, task: int) -> int | None:
    model = onnx.load_from_string(model_bytes)
    sanitized = ng.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return None
    fix_names(sanitized)
    with tempfile.TemporaryDirectory() as tmp:
        options = onnxruntime.SessionOptions()
        options.enable_profiling = True
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.log_severity_level = 3
        options.profile_file_prefix = str(pathlib.Path(tmp) / f"task{task:03d}")
        try:
            session = onnxruntime.InferenceSession(sanitized.SerializeToString(), options)
        except Exception:
            return None
        examples = ng.load_examples(task)
        batch = ng.convert_to_numpy(examples["train"][0])
        if batch is not None:
            try:
                ng.run_network(session, batch["input"])
            except Exception:
                pass
        trace_path = session.end_profiling()
        try:
            memory, params = ng.score_network(sanitized, trace_path)
        except Exception:
            return None
    if memory is None or params is None or memory < 0 or params < 0:
        return None
    return int(memory) + int(params)


def verify_all(model_bytes: bytes, task: int) -> bool:
    model = onnx.load_from_string(model_bytes)
    sanitized = ng.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return False
    fix_names(sanitized)
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    try:
        session = onnxruntime.InferenceSession(sanitized.SerializeToString(), options)
    except Exception:
        return False
    examples = ng.load_examples(task)
    for example in examples["train"] + examples["test"] + examples["arc-gen"]:
        batch = ng.convert_to_numpy(example)
        if batch is None:
            continue
        try:
            result = ng.run_network(session, batch["input"])
        except Exception:
            return False
        if not np.array_equal(result, batch["output"]):
            return False
    return True


def iter_zip_paths(roots: list[pathlib.Path]) -> list[pathlib.Path]:
    paths: list[pathlib.Path] = []
    for root in roots:
        if root.is_file() and root.suffix.lower() == ".zip":
            paths.append(root)
        elif root.is_dir():
            paths.extend(root.rglob("*.zip"))
    return sorted({p.resolve() for p in paths})


def read_task_from_zip(zf: zipfile.ZipFile, task: int) -> bytes | None:
    name = f"task{task:03d}.onnx"
    try:
        return zf.read(name)
    except KeyError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--roots",
        nargs="*",
        type=pathlib.Path,
        default=[
            NGC_ROOT / "submissions",
            NGC_ROOT / "external" / "source_review_20260709_e",
            NGC_ROOT / "external" / "new_sources_20260708",
            NGC_ROOT / "external" / "new_sources_20260708_v2",
            NGC_ROOT / "external" / "new_sources_20260708_v3",
            NGC_ROOT / "external" / "new_sources_20260708_v4",
            NGC_ROOT / "external" / "kernels_v2",
            NGC_ROOT / "external" / "kernels_v3",
            NGC_ROOT / "external" / "kernels_v4",
            NGC_ROOT / "external" / "kernels_v5",
            NGC_ROOT / "external" / "kernels_v6",
        ],
    )
    parser.add_argument("--output", type=pathlib.Path, default=OUT_CSV)
    parser.add_argument("--limit-zips", type=int)
    args = parser.parse_args()

    tasks = load_e_tasks()
    base_costs: dict[int, int] = {}
    with zipfile.ZipFile(BASE_ZIP) as base:
        for task in tasks:
            model_bytes = read_task_from_zip(base, task)
            if model_bytes is None:
                raise FileNotFoundError(f"base missing task{task:03d}.onnx")
            cost = score_cost(model_bytes, task)
            if cost is None:
                raise RuntimeError(f"could not score base task{task:03d}")
            base_costs[task] = cost

    zip_paths = [p for p in iter_zip_paths(args.roots) if p != BASE_ZIP.resolve()]
    if args.limit_zips:
        zip_paths = zip_paths[: args.limit_zips]

    rows: list[dict[str, object]] = []
    seen: set[tuple[int, str]] = set()
    scored = 0
    improved = 0
    for idx, zip_path in enumerate(zip_paths, 1):
        try:
            with zipfile.ZipFile(zip_path) as zf:
                names = set(zf.namelist())
                present = [task for task in tasks if f"task{task:03d}.onnx" in names]
                if not present:
                    continue
                for task in present:
                    model_bytes = zf.read(f"task{task:03d}.onnx")
                    digest = hashlib.sha256(model_bytes).hexdigest()
                    key = (task, digest)
                    if key in seen:
                        continue
                    seen.add(key)
                    cost = score_cost(model_bytes, task)
                    scored += 1
                    base_cost = base_costs[task]
                    status = "not_better"
                    verified = ""
                    delta_points = 0.0
                    if cost is None:
                        status = "score_error"
                    elif cost < base_cost:
                        verified_bool = verify_all(model_bytes, task)
                        verified = str(verified_bool)
                        if verified_bool:
                            status = "ok_improved"
                            improved += 1
                            delta_points = points(cost) - points(base_cost)
                        else:
                            status = "wrong_or_unverified"
                    rows.append(
                        {
                            "task": f"task{task:03d}",
                            "source_zip": str(zip_path),
                            "base_cost": base_cost,
                            "candidate_cost": "" if cost is None else cost,
                            "delta_cost": "" if cost is None else base_cost - cost,
                            "base_points": f"{points(base_cost):.9f}",
                            "candidate_points": "" if cost is None else f"{points(cost):.9f}",
                            "delta_points": f"{delta_points:.9f}",
                            "status": status,
                            "verified_all": verified,
                            "sha256": digest,
                        }
                    )
        except zipfile.BadZipFile:
            continue
        print(
            f"[{idx}/{len(zip_paths)}] {zip_path.name}: scored={scored} improved={improved}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "source_zip",
        "base_cost",
        "candidate_cost",
        "delta_cost",
        "base_points",
        "candidate_points",
        "delta_points",
        "status",
        "verified_all",
        "sha256",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")
    print(f"Rows={len(rows)} scored_unique={scored} ok_improved={improved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
