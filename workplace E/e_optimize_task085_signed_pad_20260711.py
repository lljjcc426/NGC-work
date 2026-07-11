#!/usr/bin/env python
"""Replace task085 Slice+Pad shifts with signed Pad operations."""
from __future__ import annotations

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
from onnx import numpy_helper


TASK = 85
TASK_NAME = "task085"
NGC_ROOT = pathlib.Path(r"F:\kaggle\neurogolf-2026")
BASE_ZIP = (
    NGC_ROOT
    / "submissions"
    / "submission_team_base726731_e_task233_masked_topk_20260710.zip"
)
HERE = pathlib.Path(__file__).resolve().parent
OUT_MODEL = HERE / "optimized_onnx" / "task085_signed_pad_20260711" / "task085.onnx"
OUT_CSV = HERE / "e_task085_signed_pad_20260711.csv"

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


def score_cost(model: onnx.ModelProto) -> int:
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
    return int(memory) + int(params)


def verify_all(model: onnx.ModelProto) -> dict[str, tuple[int, int]]:
    sanitized = ng.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize_model returned None")
    fix_names(sanitized)
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    session = onnxruntime.InferenceSession(sanitized.SerializeToString(), options)
    results: dict[str, tuple[int, int]] = {}
    for split in ("train", "test", "arc-gen"):
        passed = 0
        total = 0
        for example in ng.load_examples(TASK)[split]:
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
    candidate.graph.initializer.extend(
        [
            numpy_helper.from_array(np.array([1, -1], dtype=np.int64), name="shift_down"),
            numpy_helper.from_array(np.array([-1, 1], dtype=np.int64), name="shift_up"),
            numpy_helper.from_array(np.array([2], dtype=np.int64), name="shift_axis"),
        ]
    )

    slice_outputs = {"Adn_sl", "Aup_sl", "Bdn_sl", "Bup_sl"}
    kept_nodes = []
    for node in candidate.graph.node:
        if node.output and node.output[0] in slice_outputs:
            continue
        output = node.output[0] if node.output else ""
        if output in {"Adn", "Bdn", "Aup", "Bup"}:
            node.input[0] = "A" if output in {"Adn", "Aup"} else "B"
            node.input[1] = "shift_down" if output in {"Adn", "Bdn"} else "shift_up"
            while len(node.input) < 4:
                node.input.append("")
            node.input[3] = "shift_axis"
        kept_nodes.append(node)
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


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    with zipfile.ZipFile(BASE_ZIP) as archive:
        base = onnx.load_from_string(archive.read(f"{TASK_NAME}.onnx"))
    candidate, removed_initializers = build_candidate(base)
    base_cost = score_cost(base)
    candidate_cost = score_cost(candidate)
    validation = verify_all(candidate)
    passed = sum(value[0] for value in validation.values())
    total = sum(value[1] for value in validation.values())
    accepted = passed == total and candidate_cost < base_cost
    if not accepted:
        raise RuntimeError(
            f"candidate rejected: cost {base_cost}->{candidate_cost}, validation {passed}/{total}"
        )

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(candidate, OUT_MODEL)
    row = {
        "task": TASK_NAME,
        "method": "signed_pad_replaces_slice_pad",
        "base_cost": base_cost,
        "candidate_cost": candidate_cost,
        "delta_cost": base_cost - candidate_cost,
        "base_points": f"{points(base_cost):.9f}",
        "candidate_points": f"{points(candidate_cost):.9f}",
        "delta_points": f"{points(candidate_cost) - points(base_cost):.9f}",
        "train": f"{validation['train'][0]}/{validation['train'][1]}",
        "test": f"{validation['test'][0]}/{validation['test'][1]}",
        "arc_gen": f"{validation['arc-gen'][0]}/{validation['arc-gen'][1]}",
        "verified_all": True,
        "removed_nodes": 4,
        "removed_initializers": ";".join(removed_initializers),
        "model_sha256": sha256(OUT_MODEL),
    }
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
