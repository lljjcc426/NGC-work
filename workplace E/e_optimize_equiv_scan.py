#!/usr/bin/env python
"""E-team conservative equivalence optimization scan.

This script reads E-owned tasks from the shared assignment CSV, extracts the
current v244 ONNX models from the local submission zip, and tries conservative
onnxsim/onnxoptimizer rewrites. A candidate is reported only when it verifies on
all released train/test/arc-gen examples and has lower official local cost.
"""
from __future__ import annotations

import copy
import csv
import math
import pathlib
import sys
import tempfile
import zipfile

import numpy as np
import onnx
import onnxoptimizer
import onnxruntime
import onnxsim


REPO = pathlib.Path(__file__).resolve().parents[1]
NGC_ROOT = pathlib.Path(r"F:\kaggle\neurogolf-2026")
ASSIGNMENT = REPO / "assignments" / "task_assignment_400.csv"
SUBMISSION_ZIP = NGC_ROOT / "submissions" / "submission_v244_union_20260708.zip"
OUT_CSV = pathlib.Path(__file__).with_name("e_equiv_opt_scan_20260709.csv")

sys.path.insert(0, str(NGC_ROOT / "data" / "neurogolf_utils"))
import neurogolf_utils as ng  # noqa: E402

ng._NEUROGOLF_DIR = str((NGC_ROOT / "data").resolve()) + "\\"
onnxruntime.set_default_logger_severity(3)


OPTIMIZER_PASSES = [
    "eliminate_identity",
    "eliminate_nop_cast",
    "eliminate_deadend",
    "eliminate_unused_initializer",
    "eliminate_duplicate_initializer",
    "eliminate_common_subexpression",
    "eliminate_nop_reshape",
    "eliminate_nop_transpose",
    "eliminate_nop_pad",
    "eliminate_nop_concat",
    "eliminate_nop_flatten",
    "eliminate_nop_expand",
    "eliminate_nop_split",
    "fuse_consecutive_squeezes",
    "fuse_consecutive_unsqueezes",
    "fuse_consecutive_transposes",
    "fuse_consecutive_concats",
    "fuse_consecutive_reduce_unsqueeze",
    "fuse_consecutive_slices",
    "eliminate_nop_monotone_argmax",
    "eliminate_nop_dropout",
    "eliminate_if_with_const_cond",
    "eliminate_shape_gather",
    "eliminate_shape_op",
    "eliminate_slice_after_shape",
    "eliminate_consecutive_idempotent_ops",
    "eliminate_nop_with_unit",
    "extract_constant_to_initializer",
    "rewrite_where",
    "rewrite_input_dtype",
]


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


def official_cost(model: onnx.ModelProto, task: int) -> int | None:
    sanitized = ng.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return None
    fix_names(sanitized)
    with tempfile.TemporaryDirectory() as tmp:
        opts = onnxruntime.SessionOptions()
        opts.enable_profiling = True
        opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        opts.log_severity_level = 3
        opts.profile_file_prefix = str(pathlib.Path(tmp) / f"task{task:03d}")
        try:
            session = onnxruntime.InferenceSession(sanitized.SerializeToString(), opts)
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
    if memory is None or params is None:
        return None
    return int(memory) + int(params)


def verify_all(model: onnx.ModelProto, task: int) -> bool:
    sanitized = ng.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return False
    fix_names(sanitized)
    opts = onnxruntime.SessionOptions()
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    opts.log_severity_level = 3
    try:
        session = onnxruntime.InferenceSession(sanitized.SerializeToString(), opts)
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


def candidate_models(model: onnx.ModelProto):
    yield "original", model
    try:
        simplified, ok = onnxsim.simplify(
            model,
            overwrite_input_shapes=[("input", [1, 10, 30, 30])],
        )
        if ok:
            yield "onnxsim", simplified
    except Exception:
        pass
    try:
        optimized = onnxoptimizer.optimize(model, OPTIMIZER_PASSES)
        yield "onnxoptimizer", optimized
    except Exception:
        pass


def load_e_tasks() -> list[dict[str, str]]:
    with ASSIGNMENT.open(newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if row["owner"] == "E"]


def points(cost: int) -> float:
    return max(1.0, 25.0 - math.log(cost)) if cost > 0 else 25.0


def main() -> None:
    rows = []
    e_tasks = load_e_tasks()
    with zipfile.ZipFile(SUBMISSION_ZIP) as zf:
        for task_row in sorted(e_tasks, key=lambda row: float(row["points"])):
            task_name = task_row["task"]
            task = int(task_name.replace("task", ""))
            model = onnx.load_from_string(zf.read(f"{task_name}.onnx"))
            current_cost = int(float(task_row["cost"]))
            best_name = "original"
            best_cost = official_cost(model, task)
            best_verified = verify_all(model, task)

            for name, candidate in candidate_models(model):
                candidate_cost = official_cost(candidate, task)
                if candidate_cost is None:
                    continue
                if candidate_cost < (best_cost or current_cost) and verify_all(candidate, task):
                    best_name = name
                    best_cost = candidate_cost
                    best_verified = True

            improved = best_verified and best_cost is not None and best_cost < current_cost
            rows.append(
                {
                    "task": task_name,
                    "assignment_type": task_row["assignment_type"],
                    "priority_band": task_row["priority_band"],
                    "shape_class": task_row["shape_class"],
                    "current_cost": current_cost,
                    "best_method": best_name,
                    "best_cost": best_cost if best_cost is not None else "",
                    "delta_cost": current_cost - best_cost if improved else 0,
                    "current_points": task_row["points"],
                    "best_points": f"{points(best_cost):.6f}" if best_cost else "",
                    "delta_points": f"{points(best_cost) - float(task_row['points']):.6f}" if improved else "0.000000",
                    "verified_all": best_verified,
                    "improved": improved,
                }
            )
            print(
                f"{task_name}: current={current_cost} best={best_cost} "
                f"method={best_name} improved={improved}",
                flush=True,
            )

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
