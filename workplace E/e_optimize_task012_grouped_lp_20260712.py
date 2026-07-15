#!/usr/bin/env python
"""Search an exact output-direct grouped Conv rewrite for task012."""
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
from scipy.optimize import linprog  # noqa: E402


TASK = 12
TASK_NAME = "task012"
BASE_ZIP = pathlib.Path(
    r"F:\kaggle\neurogolf-2026\submissions\submission_team_high727767_e_round1_20260712.zip"
)
NGC_ROOT = pathlib.Path(r"F:\kaggle\neurogolf-2026")
HERE = pathlib.Path(__file__).resolve().parent
OUT_MODEL = HERE / "optimized_onnx" / "task012_grouped_lp_20260712" / "task012.onnx"
OUT_CSV = HERE / "e_task012_grouped_lp_20260712.csv"

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


def official_cost(model: onnx.ModelProto) -> tuple[int, int, int]:
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
        example = ng.convert_to_numpy(ng.load_examples(TASK)["train"][0])
        ng.run_network(session, example["input"])
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


def load_batches() -> list[dict[str, np.ndarray]]:
    examples = ng.load_examples(TASK)
    batches = []
    for split in ("train", "test", "arc-gen"):
        for example in examples[split]:
            batch = ng.convert_to_numpy(example)
            if batch is not None:
                batches.append(batch)
    return batches


def collect_unique_patterns(
    batches: list[dict[str, np.ndarray]], kernel: int
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    radius = kernel // 2
    tables: list[dict[bytes, int]] = [dict() for _ in range(10)]
    values: list[dict[bytes, np.ndarray]] = [dict() for _ in range(10)]
    for batch in batches:
        x = batch["input"].astype(np.uint8, copy=False)
        y = batch["output"].astype(np.uint8, copy=False)
        for channel in range(10):
            padded = np.pad(x[0, channel], radius)
            windows = np.lib.stride_tricks.sliding_window_view(
                padded, (kernel, kernel)
            ).reshape(900, kernel * kernel)
            labels = y[0, channel].reshape(900)
            packed = np.packbits(windows, axis=1)
            for index in range(900):
                key = packed[index].tobytes()
                label = int(labels[index] > 0)
                old = tables[channel].get(key)
                if old is not None and old != label:
                    return [], f"channel{channel}_conflict"
                tables[channel][key] = label
                values[channel].setdefault(key, windows[index].astype(np.float64))
    result = []
    for channel in range(10):
        features = np.stack(list(values[channel].values()))
        labels = np.array(
            [tables[channel][key] for key in values[channel]], dtype=np.int8
        )
        result.append((features, labels))
    return result, "ok"


def solve_channel(
    features: np.ndarray, labels: np.ndarray, use_bias: bool
) -> tuple[np.ndarray, float] | None:
    width = features.shape[1]
    if use_bias:
        design = np.concatenate(
            [features, np.ones((features.shape[0], 1), dtype=np.float64)], axis=1
        )
    else:
        design = features
    positive = labels == 1
    a_ub = np.concatenate([-design[positive], design[~positive]], axis=0)
    negative_features = features[~positive]
    if use_bias:
        negative_margin = -np.ones(len(negative_features), dtype=np.float64)
    else:
        negative_margin = np.where(
            np.any(negative_features != 0, axis=1), -1.0, 0.0
        )
    b_ub = np.concatenate(
        [
            -np.ones(int(positive.sum()), dtype=np.float64),
            negative_margin,
        ]
    )
    result = linprog(
        np.zeros(design.shape[1], dtype=np.float64),
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=[(None, None)] * design.shape[1],
        method="highs",
    )
    if not result.success:
        return None
    vector = result.x
    weight = vector[:width]
    bias = float(vector[-1]) if use_bias else 0.0
    scores = features @ weight + bias
    if np.any(scores[positive] <= 1e-5) or np.any(scores[~positive] > 1e-5):
        return None
    return weight.astype(np.float32), np.float32(bias).item()


def build_model(
    source: onnx.ModelProto,
    kernel: int,
    weights: np.ndarray,
    biases: np.ndarray | None,
) -> onnx.ModelProto:
    inputs = [copy.deepcopy(source.graph.input[0])]
    output = copy.deepcopy(source.graph.output[0])
    output.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    initializers = [numpy_helper.from_array(weights, name="grouped_weight")]
    node_inputs = ["input", "grouped_weight"]
    if biases is not None:
        initializers.append(numpy_helper.from_array(biases, name="grouped_bias"))
        node_inputs.append("grouped_bias")
    node = helper.make_node(
        "Conv",
        node_inputs,
        ["output"],
        name="grouped_output",
        group=10,
        kernel_shape=[kernel, kernel],
        pads=[kernel // 2] * 4,
    )
    model = helper.make_model(
        helper.make_graph([node], "task012_grouped_lp", inputs, [output], initializers),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    model.ir_version = min(source.ir_version, onnx.IR_VERSION)
    onnx.checker.check_model(model, full_check=True)
    return model


def points(cost: int) -> float:
    return max(1.0, 25.0 - math.log(cost))


def main() -> int:
    with zipfile.ZipFile(BASE_ZIP) as archive:
        source = onnx.load_from_string(archive.read(f"{TASK_NAME}.onnx"))
    base_memory, base_params, base_cost = official_cost(source)
    batches = load_batches()
    rows = []
    accepted_model = None
    accepted_row = None
    for kernel in (5, 7, 9, 11):
        patterns, pattern_status = collect_unique_patterns(batches, kernel)
        if pattern_status != "ok":
            rows.append(
                {
                    "kernel": kernel,
                    "use_bias": "",
                    "status": pattern_status,
                    "cost": "",
                    "validation": "",
                }
            )
            continue
        for use_bias in (False, True):
            channel_solutions = []
            for features, labels in patterns:
                solution = solve_channel(features, labels, use_bias)
                if solution is None:
                    break
                channel_solutions.append(solution)
            if len(channel_solutions) != 10:
                rows.append(
                    {
                        "kernel": kernel,
                        "use_bias": use_bias,
                        "status": "lp_infeasible",
                        "cost": "",
                        "validation": "",
                    }
                )
                continue
            weights = np.stack([item[0] for item in channel_solutions]).reshape(
                10, 1, kernel, kernel
            )
            biases = (
                np.array([item[1] for item in channel_solutions], dtype=np.float32)
                if use_bias
                else None
            )
            candidate = build_model(source, kernel, weights, biases)
            _, _, cost = official_cost(candidate)
            validation = verify_all(candidate)
            passed = sum(item[0] for item in validation.values())
            total = sum(item[1] for item in validation.values())
            status = "accepted" if passed == total and cost < base_cost else "rejected"
            row = {
                "kernel": kernel,
                "use_bias": use_bias,
                "status": status,
                "cost": cost,
                "validation": f"{passed}/{total}",
            }
            rows.append(row)
            if status == "accepted":
                accepted_model = candidate
                accepted_row = row
                break
        if accepted_model is not None:
            break

    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["kernel", "use_bias", "status", "cost", "validation"]
        )
        writer.writeheader()
        writer.writerows(rows)
    if accepted_model is None or accepted_row is None:
        print(rows)
        print("no accepted grouped Conv model")
        return 0

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(accepted_model, OUT_MODEL)
    model_sha = hashlib.sha256(OUT_MODEL.read_bytes()).hexdigest()
    candidate_cost = int(accepted_row["cost"])
    print(
        {
            "base_memory": base_memory,
            "base_params": base_params,
            "base_cost": base_cost,
            "candidate_cost": candidate_cost,
            "delta_cost": base_cost - candidate_cost,
            "base_points": f"{points(base_cost):.9f}",
            "candidate_points": f"{points(candidate_cost):.9f}",
            "delta_points": f"{points(candidate_cost) - points(base_cost):.9f}",
            "model_sha256": model_sha,
            **accepted_row,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
