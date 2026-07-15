from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from c_score_common import TASK_DATA_DIR, score_onnx


SHAPE = (1, 10, 30, 30)
SPATIAL_SIGNATURES = ("hw", "wh", "hh", "ww")


def _pairs(task: str) -> list[dict]:
    payload = json.loads((TASK_DATA_DIR / f"{task}.json").read_text(encoding="utf-8"))
    pairs = [item for split in ("train", "test", "arc-gen") for item in payload.get(split, [])]
    return [
        item
        for item in pairs
        if all(
            grid
            and grid[0]
            and len(grid) <= SHAPE[2]
            and len(grid[0]) <= SHAPE[3]
            for grid in (item["input"], item["output"])
        )
    ]


def _tensor(grid: list[list[int]]) -> np.ndarray:
    value = np.zeros(SHAPE, dtype=np.float32)
    for row, cells in enumerate(grid):
        for col, color in enumerate(cells):
            value[0, color, row, col] = 1.0
    return value


def _relation_factors(value: np.ndarray) -> dict[str, np.ndarray]:
    relation = value[0].astype(bool)
    occupancy = relation.any(axis=0, keepdims=True)
    diag_relation = np.diagonal(relation, axis1=1, axis2=2)
    diag_occupancy = np.diagonal(occupancy, axis1=1, axis2=2)
    return {
        "C_hw": relation,
        "C_wh": relation.transpose(0, 2, 1),
        "C_hh": np.broadcast_to(diag_relation[:, :, None], relation.shape),
        "C_ww": np.broadcast_to(diag_relation[:, None, :], relation.shape),
        "O_hw": occupancy,
        "O_wh": occupancy.transpose(0, 2, 1),
        "O_hh": np.broadcast_to(diag_occupancy[:, :, None], (1, 30, 30)),
        "O_ww": np.broadcast_to(diag_occupancy[:, None, :], (1, 30, 30)),
    }


def _factor_equation(factors: tuple[str, ...]) -> str:
    channel_labels = iter("klmopqrstuvxyz")
    terms: list[str] = []
    for factor in factors:
        kind, spatial = factor.split("_")
        channel = "c" if kind == "C" else next(channel_labels)
        terms.append(f"n{channel}{spatial}")
    return ",".join(terms) + "->nchw"


def _relational_candidates() -> list[tuple[str, ...]]:
    color = tuple(f"C_{item}" for item in SPATIAL_SIGNATURES)
    occupancy = tuple(f"O_{item}" for item in SPATIAL_SIGNATURES)
    candidates: set[tuple[str, ...]] = set()
    for count in (1, 2, 3):
        for factors in itertools.combinations_with_replacement(color + occupancy, count):
            if not any(item.startswith("C_") for item in factors):
                continue
            equation = _factor_equation(factors)
            inputs = equation.split("->", 1)[0]
            if "h" not in inputs or "w" not in inputs:
                continue
            candidates.add(factors)
    return sorted(candidates, key=lambda item: (len(item), _factor_equation(item)))


def _apply_factors(value: np.ndarray, factors: tuple[str, ...]) -> np.ndarray:
    available = _relation_factors(value)
    result = np.ones((10, 30, 30), dtype=bool)
    for factor in factors:
        result &= available[factor]
    return result[None].astype(np.float32)


def _shift(value: np.ndarray, delta: tuple[int, int, int]) -> np.ndarray:
    output = np.zeros_like(value)
    source_slices = [slice(None)]
    target_slices = [slice(None)]
    for size, shift in zip(SHAPE[1:], delta):
        if shift >= 0:
            source_slices.append(slice(0, size - shift))
            target_slices.append(slice(shift, size))
        else:
            source_slices.append(slice(-shift, size))
            target_slices.append(slice(0, size + shift))
    output[tuple(target_slices)] = value[tuple(source_slices)]
    return output


def _pad_candidates(first_input: np.ndarray, first_output: np.ndarray) -> list[tuple[int, int, int]]:
    source = np.argwhere(first_input[0] > 0)
    target = np.argwhere(first_output[0] > 0)
    if not len(source) or not len(target):
        return []
    anchor = target[0]
    candidates = {tuple(int(x) for x in anchor - item) for item in source}
    return sorted(
        delta
        for delta in candidates
        if all(-(size - 1) <= shift <= size - 1 for size, shift in zip(SHAPE[1:], delta))
        and np.array_equal(_shift(first_input, delta), first_output)
    )


def _value_info(name: str) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(SHAPE))


def _build_einsum(path: Path, factors: tuple[str, ...]) -> None:
    equation = _factor_equation(factors)
    node = helper.make_node(
        "Einsum",
        ["input"] * len(factors),
        ["output"],
        name="output",
        equation=equation,
    )
    graph = helper.make_graph([node], "zero_cost_einsum", [_value_info("input")], [_value_info("output")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, path)


def _build_pad(path: Path, delta: tuple[int, int, int]) -> None:
    begins = [0]
    ends = [0]
    for shift in delta:
        begins.append(shift if shift > 0 else shift)
        ends.append(-shift)
    node = helper.make_node(
        "Pad",
        ["input"],
        ["output"],
        name="output",
        mode="constant",
        pads=begins + ends,
        value=0.0,
    )
    graph = helper.make_graph([node], "zero_cost_pad", [_value_info("input")], [_value_info("output")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, path)


def _scan_task(task: str, parent_dir: Path, output_dir: Path, factors: list[tuple[str, ...]]) -> list[dict]:
    parent_path = parent_dir / f"{task}.onnx"
    parent_model = onnx.load(parent_path)
    if (
        len(parent_model.graph.node) == 1
        and not parent_model.graph.initializer
        and parent_model.graph.node[0].op_type != "Constant"
    ):
        return []
    examples = [(_tensor(item["input"]), _tensor(item["output"])) for item in _pairs(task)]
    if not examples:
        return []
    matches: list[tuple[str, object]] = []
    first_input, first_output = examples[0]
    for candidate in factors:
        if not np.array_equal(_apply_factors(first_input, candidate), first_output):
            continue
        if all(np.array_equal(_apply_factors(inp, candidate), out) for inp, out in examples[1:]):
            matches.append(("einsum", candidate))
    for delta in _pad_candidates(first_input, first_output):
        if all(np.array_equal(_shift(inp, delta), out) for inp, out in examples[1:]):
            matches.append(("pad", delta))

    if not matches:
        return []
    task_dir = output_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    parent = score_onnx(task, parent_path, validate_all=True)
    for index, (family, candidate) in enumerate(matches):
        path = task_dir / f"{task}_{family}_{index:03d}.onnx"
        if family == "einsum":
            _build_einsum(path, candidate)  # type: ignore[arg-type]
            detail = _factor_equation(candidate)  # type: ignore[arg-type]
        else:
            _build_pad(path, candidate)  # type: ignore[arg-type]
            detail = str(candidate)
        score = score_onnx(task, path, validate_all=True)
        accepted = bool(
            score.ok
            and score.examples_checked == score.examples_passed
            and score.cost == 0
            and parent.cost is not None
            and score.cost < parent.cost
        )
        row = {
            "task": task,
            "family": family,
            "detail": detail,
            "parent_cost": parent.cost,
            "parent_points": parent.points,
            **asdict(score),
            "accepted": accepted,
        }
        results.append(row)
        print(json.dumps(row, ensure_ascii=False, separators=(",", ":")), flush=True)
        if not accepted:
            path.unlink(missing_ok=True)
        else:
            break
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Find exact zero-cost single-node ONNX solvers.")
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", default="")
    args = parser.parse_args()

    requested = [item.strip() for item in args.tasks.split(",") if item.strip()]
    tasks = requested or [f"task{index:03d}" for index in range(1, 401)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    factors = _relational_candidates()
    accepted: list[dict] = []
    for task in tasks:
        accepted.extend(row for row in _scan_task(task, args.parent_dir, args.output_dir, factors) if row["accepted"])
    summary = {
        "tasks_scanned": len(tasks),
        "relational_candidates": len(factors),
        "accepted": len(accepted),
        "accepted_tasks": sorted({row["task"] for row in accepted}),
        "delta_points": sum(float(row["points"]) - float(row["parent_points"] or 0.0) for row in accepted),
    }
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
