#!/usr/bin/env python
"""Sweep single Cast dtype changes for current task233."""
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
import onnxruntime
from onnx import helper


NGC_ROOT = pathlib.Path(r"F:\kaggle\neurogolf-2026")
WORKPLACE = pathlib.Path(__file__).resolve().parent
BASE_ZIP = NGC_ROOT / "submissions" / "submission.zip"
OUT_DIR = WORKPLACE / "optimized_onnx" / "task233_dtype_20260710"
OUT_CSV = WORKPLACE / "e_task233_dtype_sweep_20260710.csv"

sys.path.insert(0, str(NGC_ROOT / "data" / "neurogolf_utils"))
import neurogolf_utils as ng  # noqa: E402


ng._NEUROGOLF_DIR = str((NGC_ROOT / "data").resolve()) + "\\"
onnxruntime.set_default_logger_severity(3)

DTYPES = {
    "float": onnx.TensorProto.FLOAT,
    "uint8": onnx.TensorProto.UINT8,
    "int8": onnx.TensorProto.INT8,
    "uint16": onnx.TensorProto.UINT16,
    "int16": onnx.TensorProto.INT16,
    "int32": onnx.TensorProto.INT32,
    "int64": onnx.TensorProto.INT64,
    "bool": onnx.TensorProto.BOOL,
    "float16": onnx.TensorProto.FLOAT16,
}


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


def score_cost(model: onnx.ModelProto) -> int | None:
    sanitized = ng.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return None
    fix_names(sanitized)
    with tempfile.TemporaryDirectory() as tmp:
        options = onnxruntime.SessionOptions()
        options.enable_profiling = True
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.log_severity_level = 3
        options.profile_file_prefix = str(pathlib.Path(tmp) / "task233")
        try:
            session = onnxruntime.InferenceSession(sanitized.SerializeToString(), options)
        except Exception:
            return None
        example = ng.convert_to_numpy(ng.load_examples(233)["train"][0])
        if example is not None:
            try:
                ng.run_network(session, example["input"])
            except Exception:
                return None
        trace_path = session.end_profiling()
        try:
            memory, params = ng.score_network(sanitized, trace_path)
        except Exception:
            return None
    if memory is None or params is None or memory < 0 or params < 0:
        return None
    return int(memory) + int(params)


def verify(model: onnx.ModelProto, arc_gen_limit: int | None) -> bool:
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
    examples = ng.load_examples(233)
    arc_gen = examples["arc-gen"] if arc_gen_limit is None else examples["arc-gen"][:arc_gen_limit]
    for example in examples["train"] + examples["test"] + arc_gen:
        batch = ng.convert_to_numpy(example)
        if batch is None:
            continue
        try:
            actual = ng.run_network(session, batch["input"])
        except Exception:
            return False
        if not np.array_equal(actual, batch["output"]):
            return False
    return True


def set_cast_to(node: onnx.NodeProto, dtype: int) -> None:
    del node.attribute[:]
    node.attribute.extend([helper.make_attribute("to", dtype)])


def get_cast_to(node: onnx.NodeProto) -> int | None:
    for attr in node.attribute:
        if attr.name == "to":
            return int(helper.get_attribute_value(attr))
    return None


def points(cost: int) -> float:
    return max(1.0, 25.0 - math.log(cost))


def main() -> int:
    with zipfile.ZipFile(BASE_ZIP) as zf:
        base = onnx.load_from_string(zf.read("task233.onnx"))
    base_cost = score_cost(base)
    if base_cost is None:
        raise RuntimeError("could not score base")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for node_index, node in enumerate(base.graph.node):
        if node.op_type != "Cast":
            continue
        original = get_cast_to(node)
        for label, dtype in DTYPES.items():
            if dtype == original:
                continue
            model = copy.deepcopy(base)
            set_cast_to(model.graph.node[node_index], dtype)
            cost = score_cost(model)
            status = "score_error"
            full_ok = False
            if cost is not None:
                if cost >= base_cost:
                    status = "not_better"
                elif not verify(model, arc_gen_limit=30):
                    status = "sample_wrong"
                else:
                    full_ok = verify(model, arc_gen_limit=None)
                    status = "ok_improved" if full_ok else "full_wrong"
                    if full_ok:
                        out = OUT_DIR / f"cast_n{node_index:03d}_{node.output[0]}_{label}.onnx"
                        onnx.save(model, out)
            rows.append(
                {
                    "node_index": node_index,
                    "node_output": node.output[0],
                    "original_to": original,
                    "candidate_dtype": label,
                    "base_cost": base_cost,
                    "candidate_cost": "" if cost is None else cost,
                    "delta_cost": "" if cost is None else base_cost - cost,
                    "base_points": f"{points(base_cost):.9f}",
                    "candidate_points": "" if cost is None else f"{points(cost):.9f}",
                    "delta_points": "" if cost is None else f"{points(cost) - points(base_cost):.9f}",
                    "status": status,
                    "verified_all": full_ok,
                }
            )
            print(
                f"cast n{node_index:03d} {node.output[0]} -> {label}: "
                f"status={status} cost={cost}",
                flush=True,
            )

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {OUT_CSV}")
    print(f"Improved={sum(row['status'] == 'ok_improved' for row in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
