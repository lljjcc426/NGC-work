#!/usr/bin/env python
"""Recompute the E-team task scoreboard from a submission zip."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import pathlib
import sys
import tempfile
import zipfile

import onnx
import onnxruntime


REPO = pathlib.Path(__file__).resolve().parents[1]
NGC_ROOT = pathlib.Path(r"F:\kaggle\neurogolf-2026")
ASSIGNMENT = REPO / "assignments" / "task_assignment_400.csv"
DEFAULT_ZIP = NGC_ROOT / "submissions" / "submission.zip"
OUT_CSV = pathlib.Path(__file__).with_name("e_scoreboard_20260710.csv")
OUT_JSON = pathlib.Path(__file__).with_name("e_scoreboard_summary_20260710.json")
TARGET_20_COST = 148

sys.path.insert(0, str(NGC_ROOT / "data" / "neurogolf_utils"))
import neurogolf_utils as ng  # noqa: E402


ng._NEUROGOLF_DIR = str((NGC_ROOT / "data").resolve()) + "\\"
onnxruntime.set_default_logger_severity(3)


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def points(cost: int) -> float:
    return max(1.0, 25.0 - math.log(max(1, cost)))


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


def score_cost(model_bytes: bytes, task: int) -> tuple[str, int | None, str]:
    try:
        model = onnx.load_from_string(model_bytes)
        sanitized = ng.sanitize_model(copy.deepcopy(model))
        if sanitized is None:
            return "sanitize_error", None, "sanitize_model returned None"
        fix_names(sanitized)
        with tempfile.TemporaryDirectory() as tmp:
            options = onnxruntime.SessionOptions()
            options.enable_profiling = True
            options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
            options.log_severity_level = 3
            options.profile_file_prefix = str(pathlib.Path(tmp) / f"task{task:03d}")
            session = onnxruntime.InferenceSession(sanitized.SerializeToString(), options)
            examples = ng.load_examples(task)
            batch = ng.convert_to_numpy(examples["train"][0])
            if batch is not None:
                ng.run_network(session, batch["input"])
            trace_path = session.end_profiling()
            memory, params = ng.score_network(sanitized, trace_path)
        if memory is None or params is None or memory < 0 or params < 0:
            return "score_error", None, "memory or params could not be measured"
        return "ok", int(memory) + int(params), ""
    except Exception as exc:
        return "error", None, repr(exc)


def load_e_assignments() -> list[dict[str, str]]:
    with ASSIGNMENT.open(newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row["owner"] == "E"]
    return sorted(rows, key=lambda row: int(row["task"].replace("task", "")))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=pathlib.Path, default=DEFAULT_ZIP)
    parser.add_argument("--csv", type=pathlib.Path, default=OUT_CSV)
    parser.add_argument("--json", type=pathlib.Path, default=OUT_JSON)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(args.zip) as zf:
        for assignment in load_e_assignments():
            task_name = assignment["task"]
            task = int(task_name.replace("task", ""))
            try:
                model_bytes = zf.read(f"{task_name}.onnx")
            except KeyError:
                rows.append(
                    {
                        "rank": "",
                        "task": task_name,
                        "assignment_type": assignment["assignment_type"],
                        "shape_class": assignment["shape_class"],
                        "priority_band": assignment["priority_band"],
                        "status": "missing",
                        "cost": "",
                        "points": "",
                        "above_20": False,
                        "cost_to_20_or_better": "",
                        "error": "missing from zip",
                    }
                )
                continue
            status, cost, error = score_cost(model_bytes, task)
            rows.append(
                {
                    "rank": "",
                    "task": task_name,
                    "assignment_type": assignment["assignment_type"],
                    "shape_class": assignment["shape_class"],
                    "priority_band": assignment["priority_band"],
                    "status": status,
                    "cost": "" if cost is None else cost,
                    "points": "" if cost is None else f"{points(cost):.9f}",
                    "above_20": False if cost is None else points(cost) > 20.0,
                    "cost_to_20_or_better": "" if cost is None else max(0, cost - TARGET_20_COST),
                    "error": error,
                }
            )

    scored = [row for row in rows if row["status"] == "ok"]
    scored.sort(key=lambda row: float(row["points"]))
    for rank, row in enumerate(scored, 1):
        row["rank"] = rank
    final_rows = sorted(rows, key=lambda row: (row["rank"] == "", row["rank"] or 999, row["task"]))

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "task",
        "assignment_type",
        "shape_class",
        "priority_band",
        "status",
        "cost",
        "points",
        "above_20",
        "cost_to_20_or_better",
        "error",
    ]
    with args.csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    lowest = scored[0] if scored else None
    summary = {
        "submission_zip": str(args.zip),
        "submission_sha256": sha256(args.zip),
        "scoreboard_csv": str(args.csv),
        "task_count": len(rows),
        "scored_ok": len(scored),
        "above_20_count": sum(1 for row in scored if row["above_20"]),
        "target_20_cost": TARGET_20_COST,
        "lowest_task": None if lowest is None else lowest["task"],
        "lowest_cost": None if lowest is None else lowest["cost"],
        "lowest_points": None if lowest is None else lowest["points"],
        "loop_rule": (
            "Rank E tasks by practical score ROI: meaningful expected point gain divided by "
            "implementation and full-validation time. Optimize the best verified opportunity, "
            "update the scoreboard, then rescan and continue."
        ),
    }
    args.json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
