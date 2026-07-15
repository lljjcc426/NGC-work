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
DEFAULT_OUTPUT = OWNED_DIR / "debug" / "task243_fused_lowrank_selector.onnx"
DEFAULT_EVIDENCE = OWNED_DIR / "debug" / "task243_fused_lowrank_selector_evidence.json"
TASK = "task243"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def equation(node: onnx.NodeProto) -> str:
    attr = next(attr for attr in node.attribute if attr.name == "equation")
    return helper.get_attribute_value(attr).decode("ascii")


def selector_factors() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    branch_rank = np.zeros((2, 4), dtype=np.float32)
    rank_output = np.zeros((4, 10), dtype=np.float32)
    rank_input = np.zeros((4, 10), dtype=np.float32)

    # Three terms give 0.5 - (output_color - input_color)^2 for branch 0.
    branch_rank[0, :3] = 1.0
    colors = np.arange(10, dtype=np.float32)
    rank_output[0] = 1.0
    rank_input[0] = 0.5 - colors * colors
    rank_output[1] = colors
    rank_input[1] = 2.0 * colors
    rank_output[2] = colors * colors
    rank_input[2] = -1.0

    # Branch 1 starts only at blue and turns every reached black cell blue.
    branch_rank[1, 3] = 1.0
    rank_output[3, 0] = -2.0
    rank_output[3, 1] = 2.0
    rank_input[3, 1] = 1.0
    return branch_rank, rank_output, rank_input


def build_candidate(parent_path: Path, output_path: Path) -> Path:
    model = onnx.load(str(parent_path))
    if len(model.graph.node) != 2 or any(node.op_type != "Einsum" for node in model.graph.node):
        raise ValueError("expected the two-Einsum task243 parent")

    terminal = model.graph.node[1]
    lhs, rhs = equation(terminal).split("->")
    terms = lhs.split(",")
    inputs = list(terminal.input)
    if len(terms) != 92 or len(inputs) != 92:
        raise ValueError(f"unexpected terminal arity: {len(terms)} terms, {len(inputs)} inputs")
    if terms[1] != "sb" or terms[-2:] != ["...wRC", "svw"]:
        raise ValueError("unexpected terminal selector/tail layout")
    if inputs[1] != "Qb2" or inputs[-2:] != ["input", "T2"]:
        raise ValueError("unexpected terminal selector/tail inputs")

    new_terms = [terms[0], "sw", "wv", "wb", *terms[2:-2]]
    new_inputs = [inputs[0], "QF_s", "QF_v", "QF_b", *inputs[2:-2]]
    new_terminal = helper.make_node(
        "Einsum",
        new_inputs,
        list(terminal.output),
        name=terminal.name,
        equation=",".join(new_terms) + "->" + rhs,
    )
    terminal.CopyFrom(new_terminal)

    kept = [init for init in model.graph.initializer if init.name not in {"Qb2", "T2"}]
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(kept)
    q_s, q_v, q_b = selector_factors()
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(q_s, name="QF_s"),
            numpy_helper.from_array(q_v, name="QF_v"),
            numpy_helper.from_array(q_b, name="QF_b"),
        ]
    )

    onnx.checker.check_model(model, full_check=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    return output_path


def make_session(path: Path, official_utils) -> ort.InferenceSession:
    model = official_utils.sanitize_model(onnx.load(str(path)))
    if model is None:
        raise RuntimeError(f"official sanitizer rejected {path}")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model.SerializeToString(),
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


def random_exact_compare(
    parent_path: Path,
    candidate_path: Path,
    official_utils,
    cases: int,
    seed: int,
) -> dict[str, int]:
    parent = make_session(parent_path, official_utils)
    candidate = make_session(candidate_path, official_utils)
    rng = np.random.default_rng(seed)
    exact = 0
    for _ in range(cases):
        tensor = random_input(rng)
        parent_output = parent.run(["output"], {"input": tensor})[0] > 0
        candidate_output = candidate.run(["output"], {"input": tensor})[0] > 0
        exact += int(np.array_equal(parent_output, candidate_output))
    return {"seed": seed, "cases": cases, "exact_threshold_outputs": exact}


def graph_facts(path: Path) -> dict[str, object]:
    model = onnx.load(str(path))
    return {
        "nodes": len(model.graph.node),
        "op_types": [node.op_type for node in model.graph.node],
        "initializer_params": sum(int(np.prod(init.dims)) for init in model.graph.initializer),
        "terminal_operands": len(model.graph.node[1].input),
        "terminal_equation": equation(model.graph.node[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--random-cases", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2433902)
    args = parser.parse_args()

    sys.path.insert(0, str(COMMON_DIR))
    from c_score_common import load_official_utils, score_onnx

    candidate_path = build_candidate(args.parent, args.output)
    parent_score = score_onnx(TASK, args.parent, validate_all=True)
    candidate_score = score_onnx(TASK, candidate_path, validate_all=True)
    official_utils = load_official_utils()
    random_result = random_exact_compare(
        args.parent,
        candidate_path,
        official_utils,
        args.random_cases,
        args.seed,
    )

    evidence = {
        "task": TASK,
        "rewrite": "fuse terminal color input and T2 into rank-4 seed/output selector",
        "proof_scores": {
            "base": "0.5 - (output_color - input_color)^2",
            "flood": "blue_seed * (-2 for black, +2 for blue)",
        },
        "parent": asdict(parent_score),
        "candidate": asdict(candidate_score),
        "parent_sha256": sha256_file(args.parent),
        "candidate_sha256": sha256_file(candidate_path),
        "parent_graph": graph_facts(args.parent),
        "candidate_graph": graph_facts(candidate_path),
        "random_parent_exact": random_result,
    }
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
