from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
TASK = "task074"
PARENT = (
    REPO
    / "workplace C"
    / "artifacts"
    / "full400_round37b_isolate_dilated_187"
    / "onnx"
    / "task074.onnx"
)
COMMON = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"
OUT_DIR = HERE.parent / "onnx"
REPORT = HERE.parent / "validation.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parent_arrays() -> dict[str, np.ndarray]:
    model = onnx.load(PARENT)
    return {
        item.name: numpy_helper.to_array(item).copy()
        for item in model.graph.initializer
    }


def make_model(
    nodes: list[onnx.NodeProto],
    output_type: int,
    initializers: list[onnx.TensorProto] | None = None,
    sparse_initializers: list[onnx.SparseTensorProto] | None = None,
) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        TASK,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", output_type, [1, 10, 30, 30])],
        initializer=initializers or [],
    )
    graph.sparse_initializer.extend(sparse_initializers or [])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
        producer_name="task074_agent_round39",
    )
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model


def save_model(model: onnx.ModelProto, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    onnx.save(model, path)
    onnx.checker.check_model(onnx.load(path), full_check=True)
    return path


def build_conservative(arrays: dict[str, np.ndarray]) -> Path:
    nodes = [
        helper.make_node("Conv", ["input", "color_w"], ["color_f"], kernel_shape=[1, 1]),
        helper.make_node(
            "ScatterND",
            ["base", "oidx", "color_f"],
            ["orbit_f"],
            reduction="max",
        ),
        helper.make_node("GatherND", ["orbit_f", "oidx"], ["grid_f"]),
        helper.make_node("Equal", ["grid_f", "chan_f"], ["output"]),
    ]
    initializers = [
        numpy_helper.from_array(arrays["color_w"].astype(np.float32), "color_w"),
        numpy_helper.from_array(arrays["base"].astype(np.float32), "base"),
        numpy_helper.from_array(arrays["oidx"].astype(np.int64), "oidx"),
        numpy_helper.from_array(arrays["chan"].astype(np.float32), "chan_f"),
    ]
    return save_model(
        make_model(nodes, TensorProto.BOOL, initializers=initializers),
        "task074_conservative_no_cast.onnx",
    )


def build_terminal_onehot(arrays: dict[str, np.ndarray]) -> Path:
    nodes = [
        helper.make_node("Conv", ["input", "color_w"], ["color_f"], kernel_shape=[1, 1]),
        helper.make_node(
            "ScatterND",
            ["base", "oidx_scatter", "color_f"],
            ["orbit_f"],
            reduction="max",
        ),
        helper.make_node("Cast", ["orbit_f"], ["orbit_u8"], to=TensorProto.UINT8),
        helper.make_node("GatherND", ["orbit_u8", "oidx_grid"], ["grid_u8"]),
        helper.make_node(
            "OneHot",
            ["grid_u8", "depth", "onehot_values"],
            ["output"],
            axis=1,
        ),
    ]
    initializers = [
        numpy_helper.from_array(arrays["color_w"].astype(np.float32), "color_w"),
        numpy_helper.from_array(arrays["base"].astype(np.float32), "base"),
        numpy_helper.from_array(arrays["oidx"].astype(np.int64), "oidx_scatter"),
        numpy_helper.from_array(
            arrays["oidx"].reshape(1, 30, 30, 1).astype(np.int64),
            "oidx_grid",
        ),
        numpy_helper.from_array(np.asarray(10, dtype=np.int64), "depth"),
        numpy_helper.from_array(np.asarray([False, True]), "onehot_values"),
    ]
    return save_model(
        make_model(nodes, TensorProto.BOOL, initializers=initializers),
        "task074_conservative_terminal_onehot.onnx",
    )


def build_shared_weight_conservative(arrays: dict[str, np.ndarray]) -> Path:
    color_w = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    color_w[0, 9, 0, 0] = -1.0
    nodes = [
        helper.make_node("Conv", ["input", "color_w"], ["color_f"], kernel_shape=[1, 1]),
        helper.make_node(
            "ScatterND",
            ["base", "oidx", "color_f"],
            ["orbit_f"],
            reduction="max",
        ),
        helper.make_node("GatherND", ["orbit_f", "oidx"], ["grid_f"]),
        helper.make_node("Equal", ["grid_f", "color_w"], ["output"]),
    ]
    initializers = [
        numpy_helper.from_array(color_w, "color_w"),
        numpy_helper.from_array(arrays["base"].astype(np.float32), "base"),
        numpy_helper.from_array(arrays["oidx"].astype(np.int64), "oidx"),
    ]
    return save_model(
        make_model(nodes, TensorProto.BOOL, initializers=initializers),
        "task074_conservative_shared_weight.onnx",
    )


def build_direct_einsum(arrays: dict[str, np.ndarray]) -> Path:
    oidx = arrays["oidx"].reshape(30, 30).astype(np.int64)
    orbit_count = int(oidx.max()) + 1
    incidence = np.zeros((orbit_count, 30, 30), dtype=np.float32)
    rows, cols = np.indices((30, 30))
    incidence[oidx, rows, cols] = 1.0

    priority = np.zeros((10, 10), dtype=np.float32)
    priority[0, [0, 9]] = 1.0
    priority[0, 1:9] = -9.0
    for color in range(1, 9):
        priority[color, color] = 1.0
        priority[color, color + 1 : 9] = -9.0

    initializers = [
        numpy_helper.from_array(incidence, "orbit_incidence"),
        numpy_helper.from_array(priority, "priority"),
    ]
    node = helper.make_node(
        "Einsum",
        ["input", "orbit_incidence", "orbit_incidence", "priority"],
        ["output"],
        equation="ndab,gab,gij,cd->ncij",
    )
    return save_model(
        make_model([node], TensorProto.FLOAT, initializers=initializers),
        "task074_direct_dense_einsum.onnx",
    )


def session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        path.read_bytes(),
        options,
        providers=["CPUExecutionProvider"],
    )


def threshold_output(sess: ort.InferenceSession, value: np.ndarray) -> np.ndarray:
    return sess.run(["output"], {"input": value})[0] > 0


def random_exact(paths: list[Path], cases: int, seed: int = 74039) -> dict:
    rng = np.random.default_rng(seed)
    parent_session = session(PARENT)
    candidate_sessions = {path.name: session(path) for path in paths}
    mismatches = {path.name: 0 for path in paths}
    mismatched_cells = {path.name: 0 for path in paths}

    for _ in range(cases):
        colors = rng.integers(0, 10, size=(30, 30))
        value = np.zeros((1, 10, 30, 30), dtype=np.float32)
        value[0, colors, *np.indices((30, 30))] = 1.0
        expected = threshold_output(parent_session, value)
        for name, candidate_session in candidate_sessions.items():
            actual = threshold_output(candidate_session, value)
            difference = actual != expected
            if np.any(difference):
                mismatches[name] += 1
                mismatched_cells[name] += int(np.sum(difference))

    return {
        "seed": seed,
        "cases": cases,
        "input_domain": "uniform random one-hot colors 0..9 on every 30x30 cell",
        "mismatched_cases": mismatches,
        "mismatched_cells": mismatched_cells,
    }


def graph_summary(path: Path) -> dict:
    model = onnx.load(path)
    dense_params = sum(math.prod(item.dims) for item in model.graph.initializer)
    sparse_params = sum(
        math.prod(item.values.dims) for item in model.graph.sparse_initializer
    )
    return {
        "path": str(path),
        "sha256": sha256(path),
        "file_size": path.stat().st_size,
        "nodes": [node.op_type for node in model.graph.node],
        "dense_initializer_elements": dense_params,
        "sparse_initializer_values": sparse_params,
        "checker_full_check": True,
    }


def official_scores(paths: list[Path]) -> list[dict]:
    sys.path.insert(0, str(COMMON))
    from c_score_common import score_onnx

    return [asdict(score_onnx(TASK, path, validate_all=True)) for path in paths]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-official", action="store_true")
    parser.add_argument("--random-cases", type=int, default=256)
    args = parser.parse_args()

    arrays = parent_arrays()
    candidates = [
        build_terminal_onehot(arrays),
        build_conservative(arrays),
        build_shared_weight_conservative(arrays),
        build_direct_einsum(arrays),
    ]
    checked_paths = [PARENT, *candidates]
    report = {
        "task": TASK,
        "parent": graph_summary(PARENT),
        "candidates": [graph_summary(path) for path in candidates],
        "orbit_evidence": {
            "orbit_count": int(arrays["oidx"].max()) + 1,
            "cell_count": int(arrays["oidx"].size),
            "max_orbit_size": int(
                np.unique(arrays["oidx"], return_counts=True)[1].max()
            ),
            "direct_equation": "ndab,gab,gij,cd->ncij",
            "negative_priority_weight": -9,
            "proof": (
                "For a one-hot cell grid, each direct output score sums +1 for its "
                "color over the target orbit and -9 for every higher color. Every "
                "orbit has at most 8 cells, so a higher color makes the score negative; "
                "otherwise the unique highest present color is positive. Color 9 is "
                "mapped with color 0 exactly as in the parent Conv weights."
            ),
        },
        "random_exact": random_exact(candidates, args.random_cases),
        "official_scores": [] if args.skip_official else official_scores(checked_paths),
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
