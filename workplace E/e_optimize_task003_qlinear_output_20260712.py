#!/usr/bin/env python
"""Optimize task003 by folding inverse-channel creation into QLinearConv."""
from __future__ import annotations

import copy
import csv
import hashlib
import math
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
from onnx import helper, numpy_helper  # noqa: E402


TASK = 3
TASK_NAME = "task003"
BASE_ZIP = pathlib.Path(r"F:\kaggle\submission (1).zip")
NGC_ROOT = pathlib.Path(r"F:\kaggle\neurogolf-2026")
HERE = pathlib.Path(__file__).resolve().parent
OUT_MODEL = HERE / "optimized_onnx" / "task003_qlinear_output_20260712" / "task003.onnx"
OUT_CSV = HERE / "e_task003_qlinear_output_20260712.csv"

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


def score_cost(model: onnx.ModelProto) -> tuple[int, int, int]:
    sanitized = ng.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize_model returned None")
    fix_names(sanitized)
    with tempfile.TemporaryDirectory() as tmp:
        options = onnxruntime.SessionOptions()
        options.enable_profiling = True
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.log_severity_level = 3
        options.profile_file_prefix = str(pathlib.Path(tmp) / TASK_NAME)
        session = onnxruntime.InferenceSession(sanitized.SerializeToString(), options)
        examples = ng.load_examples(TASK)
        batch = ng.convert_to_numpy(examples["train"][0])
        ng.run_network(session, batch["input"])
        trace_path = session.end_profiling()
        memory, params = ng.score_network(sanitized, trace_path)
    if memory is None or params is None:
        raise RuntimeError("official scorer returned no result")
    return int(memory), int(params), int(memory) + int(params)


def verify_all(model: onnx.ModelProto) -> dict[str, tuple[int, int]]:
    sanitized = ng.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize_model returned None")
    fix_names(sanitized)
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    session = onnxruntime.InferenceSession(sanitized.SerializeToString(), options)
    examples = ng.load_examples(TASK)
    results: dict[str, tuple[int, int]] = {}
    for split in ("train", "test", "arc-gen"):
        passed = 0
        total = 0
        for example in examples[split]:
            batch = ng.convert_to_numpy(example)
            if batch is None:
                continue
            total += 1
            output = ng.run_network(session, batch["input"])
            passed += int(np.array_equal(output, batch["output"]))
        results[split] = (passed, total)
    return results


def build_candidate(base: onnx.ModelProto) -> tuple[onnx.ModelProto, list[str]]:
    candidate = copy.deepcopy(base)
    output_node = next(node for node in candidate.graph.node if "output" in node.output)
    kept_nodes = [
        node
        for node in candidate.graph.node
        if node.output and node.output[0] not in {"ch0_valid", "ch0ch2", "output"}
    ]

    weights = np.zeros((2, 1, 1, 1), dtype=np.int8)
    weights[0, 0, 0, 0] = -1
    weights[1, 0, 0, 0] = 1
    bias = np.zeros(2, dtype=np.int32)
    bias[0] = 1
    candidate.graph.initializer.extend(
        [
            numpy_helper.from_array(np.array(1.0, dtype=np.float32), name="q_scale"),
            numpy_helper.from_array(np.array(0, dtype=np.uint8), name="q_zero_u8"),
            numpy_helper.from_array(np.array(0, dtype=np.int8), name="q_zero_i8"),
            numpy_helper.from_array(weights, name="q_weight"),
            numpy_helper.from_array(bias, name="q_bias"),
        ]
    )
    kept_nodes.append(
        helper.make_node(
            "QLinearConv",
            [
                "ch2_valid",
                "q_scale",
                "q_zero_u8",
                "q_weight",
                "q_scale",
                "q_zero_i8",
                "q_scale",
                "q_zero_u8",
                "q_bias",
            ],
            ["ch0ch2"],
            name="qlinear_inverse_channels",
            kernel_shape=[1, 1],
        )
    )
    kept_nodes.append(output_node)
    del candidate.graph.node[:]
    candidate.graph.node.extend(kept_nodes)

    used = {name for node in candidate.graph.node for name in node.input if name}
    removed = [init.name for init in candidate.graph.initializer if init.name not in used]
    kept_initializers = [init for init in candidate.graph.initializer if init.name in used]
    del candidate.graph.initializer[:]
    candidate.graph.initializer.extend(kept_initializers)
    onnx.checker.check_model(candidate, full_check=True)
    return candidate, removed


def points(cost: int) -> float:
    return max(1.0, 25.0 - math.log(cost))


def main() -> int:
    with zipfile.ZipFile(BASE_ZIP) as archive:
        base = onnx.load_from_string(archive.read(f"{TASK_NAME}.onnx"))
    candidate, removed_initializers = build_candidate(base)
    base_memory, base_params, base_cost = score_cost(base)
    candidate_memory, candidate_params, candidate_cost = score_cost(candidate)
    validation = verify_all(candidate)
    passed = sum(result[0] for result in validation.values())
    total = sum(result[1] for result in validation.values())
    if passed != total or candidate_cost >= base_cost:
        raise RuntimeError(
            f"candidate rejected: cost {base_cost}->{candidate_cost}, validation {passed}/{total}"
        )

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(candidate, OUT_MODEL)
    model_sha256 = hashlib.sha256(OUT_MODEL.read_bytes()).hexdigest()
    row = {
        "task": TASK_NAME,
        "method": "qlinear_fuses_inverse_and_concat",
        "base_memory": base_memory,
        "base_params": base_params,
        "base_cost": base_cost,
        "candidate_memory": candidate_memory,
        "candidate_params": candidate_params,
        "candidate_cost": candidate_cost,
        "delta_cost": base_cost - candidate_cost,
        "base_points": f"{points(base_cost):.9f}",
        "candidate_points": f"{points(candidate_cost):.9f}",
        "delta_points": f"{points(candidate_cost) - points(base_cost):.9f}",
        "train": f"{validation['train'][0]}/{validation['train'][1]}",
        "test": f"{validation['test'][0]}/{validation['test'][1]}",
        "arc_gen": f"{validation['arc-gen'][0]}/{validation['arc-gen'][1]}",
        "verified_all": True,
        "removed_nodes": 1,
        "removed_initializers": ";".join(removed_initializers),
        "model_sha256": model_sha256,
    }
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
