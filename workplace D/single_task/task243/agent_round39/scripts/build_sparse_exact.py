from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper


REPO_ROOT = Path(__file__).resolve().parents[5]
OWNED_DIR = Path(__file__).resolve().parents[1]
COMMON_DIR = REPO_ROOT / "workplace C" / "neurogolf-2026-work" / "scripts"
DEFAULT_PARENT = (
    REPO_ROOT
    / "workplace C"
    / "artifacts"
    / "full400_round37b_isolate_dilated_187"
    / "onnx"
    / "task243.onnx"
)
DEFAULT_OUTPUT = OWNED_DIR / "debug" / "task243_sparse_exact.onnx"
DEFAULT_EVIDENCE = OWNED_DIR / "debug" / "task243_sparse_exact_evidence.json"
TASK = "task243"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_sparse_initializer(array: np.ndarray, name: str) -> onnx.SparseTensorProto:
    coordinates = np.argwhere(array != 0).astype(np.int64, copy=False)
    values_array = array[tuple(coordinates.T)]
    values = numpy_helper.from_array(values_array, name=name)
    indices = numpy_helper.from_array(coordinates)
    return helper.make_sparse_tensor(values, indices, array.shape)


def build_sparse_exact(parent_path: Path, output_path: Path) -> tuple[Path, dict[str, int]]:
    model = onnx.load(str(parent_path))
    if len(model.graph.node) != 2 or any(node.op_type != "Einsum" for node in model.graph.node):
        raise ValueError("expected the two-Einsum task243 parent")

    arrays = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    expected = {"PSEL2", "Cr", "Qb2", "S2", "T2"}
    if set(arrays) != expected:
        raise ValueError(f"unexpected initializer set: {sorted(arrays)}")

    # The official sanitizer assigns safe_name_N in first-use order after dense
    # initializers. Matching that order keeps sparse initializer names connected,
    # because the sanitizer intentionally leaves SparseTensorProto names unchanged.
    rename = {
        "PSEL2": "safe_name_0",
        "Cr": "safe_name_1",
        "Qb2": "safe_name_3",
        "S2": "safe_name_4",
        "T2": "safe_name_5",
    }
    for node in model.graph.node:
        for index, old_name in enumerate(node.input):
            if old_name in rename:
                node.input[index] = rename[old_name]

    model.graph.ClearField("initializer")
    model.graph.ClearField("sparse_initializer")
    nnz: dict[str, int] = {}
    for old_name in ("PSEL2", "Cr", "Qb2", "S2", "T2"):
        array = np.asarray(arrays[old_name])
        sparse = make_sparse_initializer(array, rename[old_name])
        model.graph.sparse_initializer.append(sparse)
        nnz[old_name] = int(np.count_nonzero(array))

    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    return output_path, nnz


def make_session(path: Path, official_utils) -> ort.InferenceSession:
    model = onnx.load(str(path))
    sanitized = official_utils.sanitize_model(model)
    if sanitized is None:
        raise RuntimeError(f"official sanitizer rejected {path}")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        sanitized.SerializeToString(),
        options,
        providers=["CPUExecutionProvider"],
    )


def random_input(rng: np.random.Generator) -> np.ndarray:
    size = int(rng.integers(12, 19))
    black_probability = float(rng.uniform(0.42, 0.58))
    colors = np.empty((size, size), dtype=np.int64)
    black = rng.random((size, size)) < black_probability
    colors[black] = 0
    colors[~black] = rng.integers(1, 10, size=int(np.count_nonzero(~black)))
    if not np.any(colors == 1):
        row, col = rng.integers(0, size, size=2)
        colors[row, col] = 1

    tensor = np.zeros((1, 10, 30, 30), dtype=np.float32)
    rows, cols = np.indices((size, size))
    tensor[0, colors, rows, cols] = 1.0
    return tensor


def compare_random(
    parent_path: Path,
    candidate_path: Path,
    official_utils,
    cases: int,
    seed: int,
) -> dict[str, int | float]:
    parent_session = make_session(parent_path, official_utils)
    candidate_session = make_session(candidate_path, official_utils)
    rng = np.random.default_rng(seed)
    raw_exact = 0
    threshold_exact = 0
    max_abs_error = 0.0

    for _ in range(cases):
        tensor = random_input(rng)
        parent_raw = parent_session.run(["output"], {"input": tensor})[0]
        candidate_raw = candidate_session.run(["output"], {"input": tensor})[0]
        if np.array_equal(parent_raw, candidate_raw):
            raw_exact += 1
        if np.array_equal(parent_raw > 0, candidate_raw > 0):
            threshold_exact += 1
        max_abs_error = max(
            max_abs_error,
            float(np.max(np.abs(parent_raw - candidate_raw))),
        )

    return {
        "seed": seed,
        "cases": cases,
        "raw_exact": raw_exact,
        "threshold_exact": threshold_exact,
        "max_abs_error": max_abs_error,
    }


def graph_facts(path: Path) -> dict[str, object]:
    model = onnx.load(str(path))
    return {
        "nodes": len(model.graph.node),
        "op_types": [node.op_type for node in model.graph.node],
        "dense_initializers": len(model.graph.initializer),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "sparse_value_params": sum(
            int(np.prod(sparse.values.dims)) for sparse in model.graph.sparse_initializer
        ),
        "equations": [
            helper.get_attribute_value(next(attr for attr in node.attribute if attr.name == "equation")).decode()
            for node in model.graph.node
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--random-cases", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2433901)
    args = parser.parse_args()

    sys.path.insert(0, str(COMMON_DIR))
    from c_score_common import load_official_utils, score_onnx

    candidate_path, nnz = build_sparse_exact(args.parent, args.output)
    parent_score = score_onnx(TASK, args.parent, validate_all=True)
    candidate_score = score_onnx(TASK, candidate_path, validate_all=True)
    official_utils = load_official_utils()
    random_result = compare_random(
        args.parent,
        candidate_path,
        official_utils,
        args.random_cases,
        args.seed,
    )

    evidence = {
        "task": TASK,
        "rewrite": "dense initializers to exact SparseTensorProto initializers",
        "parent": asdict(parent_score),
        "candidate": asdict(candidate_score),
        "parent_sha256": sha256_file(args.parent),
        "candidate_sha256": sha256_file(candidate_path),
        "initializer_nnz": nnz,
        "parent_graph": graph_facts(args.parent),
        "candidate_graph": graph_facts(candidate_path),
        "random_parent_exact": random_result,
    }
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
