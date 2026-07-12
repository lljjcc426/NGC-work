#!/usr/bin/env python
"""Fold task011 row/column count offsets into the two source Einsums."""
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
from onnx import numpy_helper  # noqa: E402


TASK = 11
TASK_NAME = "task011"
BASE_ZIP = pathlib.Path(
    r"F:\kaggle\neurogolf-2026\submissions\submission_team_high_e_task003_qlinear_20260712.zip"
)
NGC_ROOT = pathlib.Path(r"F:\kaggle\neurogolf-2026")
HERE = pathlib.Path(__file__).resolve().parent
OUT_MODEL = HERE / "optimized_onnx" / "task011_affine_counts_20260712" / "task011.onnx"
OUT_CSV = HERE / "e_task011_affine_counts_20260712.csv"

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


def replace_initializer(
    model: onnx.ModelProto, name: str, value: np.ndarray
) -> None:
    initializer = next(item for item in model.graph.initializer if item.name == name)
    initializer.CopyFrom(numpy_helper.from_array(value, name=name))


def build_candidate(base: onnx.ModelProto) -> tuple[onnx.ModelProto, list[str]]:
    candidate = copy.deepcopy(base)
    channel_affine = np.full(10, -4.0, dtype=np.float32)
    channel_affine[0] = 7.0
    replace_initializer(candidate, "e0", channel_affine)

    expand = numpy_helper.to_array(
        next(item for item in candidate.graph.initializer if item.name == "EXP4")
    ).copy()
    expand[3, 3] = -1.0
    expand[7, 3] = -1.0
    replace_initializer(candidate, "EXP4", expand.astype(np.float32))

    candidate.graph.node[0].output[0] = "wr4"
    candidate.graph.node[1].output[0] = "wc4"
    kept_nodes = [candidate.graph.node[0], candidate.graph.node[1], candidate.graph.node[4]]
    del candidate.graph.node[:]
    candidate.graph.node.extend(kept_nodes)

    used = {name for node in candidate.graph.node for name in node.input if name}
    removed = [init.name for init in candidate.graph.initializer if init.name not in used]
    kept_initializers = [init for init in candidate.graph.initializer if init.name in used]
    del candidate.graph.initializer[:]
    candidate.graph.initializer.extend(kept_initializers)
    del candidate.graph.value_info[:]
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
        "method": "fold_sub_offsets_into_affine_channel_counts",
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
        "removed_nodes": 2,
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
