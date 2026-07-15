from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper


HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
TASK_ROOT = REPO / "neurogolf_400_tasks" / "tasks"
DEFAULT_OUTPUT = REPO / "workplace C" / "artifacts" / "zero_cost_einsum_search"


def encode(grid: list[list[int]], size: int) -> np.ndarray:
    value = np.zeros((1, 10, size, size), dtype=np.float32)
    for row, cells in enumerate(grid):
        for col, color in enumerate(cells):
            value[0, color, row, col] = 1.0
    return value


def examples(task_path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    payload = json.loads(task_path.read_text(encoding="utf-8"))
    result = []
    for split in ("train", "test", "arc-gen"):
        for item in payload.get(split, []):
            input_grid = item["input"]
            output_grid = item["output"]
            input_shape = (len(input_grid), len(input_grid[0]) if input_grid else 0)
            output_shape = (len(output_grid), len(output_grid[0]) if output_grid else 0)
            if max(*input_shape, *output_shape) > 30:
                continue
            size = max(*input_shape, *output_shape)
            result.append((encode(input_grid, size), encode(output_grid, size)))
    return result


def mask_equations() -> list[str]:
    equations = set()
    for first_spatial in (("h", "w"), ("w", "h")):
        first = "nc" + "".join(first_spatial)
        for channel in ("c", "k"):
            for row in ("h", "w", "r"):
                for col in ("h", "w", "s"):
                    second = "n" + channel + row + col
                    equations.add(f"{first},{second}->nchw")
    return sorted(equations)


def general_equations() -> list[str]:
    operands = []
    for channel in ("c", "k"):
        for row in ("h", "w", "r", "s"):
            for col in ("h", "w", "r", "s"):
                operands.append("n" + channel + row + col)
    equations = set()
    for left, right in itertools.combinations_with_replacement(operands, 2):
        labels = set(left + right)
        if {"n", "c", "h", "w"} <= labels:
            equations.add(f"{left},{right}->nchw")
    return sorted(equations)


def triple_mask_equations() -> list[str]:
    masks = []
    for channel in ("c", "k"):
        for row in ("h", "w", "r", "s"):
            for col in ("h", "w", "r", "s"):
                masks.append("n" + channel + row + col)
    equations = set()
    for source in ("nchw", "ncwh"):
        for left, right in itertools.combinations_with_replacement(masks, 2):
            equations.add(f"{source},{left},{right}->nchw")
    return sorted(equations)


def triple_relayout_equations() -> list[str]:
    sources = [
        "nc" + row + col
        for row in ("h", "w", "r", "s")
        for col in ("h", "w", "r", "s")
    ]
    masks = [
        "nk" + row + col
        for row in ("h", "w", "r", "s")
        for col in ("h", "w", "r", "s")
    ]
    equations = set()
    required = {"n", "c", "h", "w"}
    for source in sources:
        for left, right in itertools.combinations_with_replacement(masks, 2):
            if required <= set(source + left + right):
                equations.add(f"{source},{left},{right}->nchw")
    return sorted(equations)


def triple_general_equations() -> list[str]:
    """Enumerate canonical three-atom conjunctive queries over one-hot grids."""
    atoms = [
        (channel, row, col)
        for channel in ("c", "k", "l")
        for row in ("h", "w", "r", "s", "t")
        for col in ("h", "w", "r", "s", "t")
    ]

    def canonical(combo: tuple[tuple[str, str, str], ...]) -> str:
        channel_map: dict[str, str] = {}
        spatial_map: dict[str, str] = {}
        channel_names = iter("klmnopq")
        spatial_names = iter("rstuvwxyzABCDEFG")
        operands = []
        for atom in sorted(combo):
            labels = []
            for axis, label in enumerate(atom):
                if axis == 0:
                    if label == "c":
                        labels.append(label)
                    else:
                        if label not in channel_map:
                            channel_map[label] = next(channel_names)
                        labels.append(channel_map[label])
                elif label in {"h", "w"}:
                    labels.append(label)
                else:
                    if label not in spatial_map:
                        spatial_map[label] = next(spatial_names)
                    labels.append(spatial_map[label])
            operands.append("n" + "".join(labels))
        return ",".join(sorted(operands)) + "->nchw"

    equations = {
        canonical(combo)
        for combo in itertools.combinations_with_replacement(atoms, 3)
    }
    return sorted(
        equation
        for equation in equations
        if all(label in equation.split("->", 1)[0] for label in "chw")
    )


def matches(equation: str, pairs: list[tuple[np.ndarray, np.ndarray]]) -> bool:
    operand_count = equation.split("->", 1)[0].count(",") + 1
    for source, target in pairs:
        try:
            predicted = np.einsum(
                equation, *(source for _ in range(operand_count)), optimize="greedy"
            ) > 0
        except (ValueError, MemoryError):
            return False
        if not np.array_equal(predicted, target):
            return False
    return True


def build(equation: str, output_path: Path) -> None:
    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 10, 30, 30]
    )
    output_info = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 10, 30, 30]
    )
    operand_count = equation.split("->", 1)[0].count(",") + 1
    node = helper.make_node(
        "Einsum", ["input"] * operand_count, ["output"], equation=equation
    )
    graph = helper.make_graph([node], "zero_cost_einsum", [input_info], [output_info])
    model = helper.make_model(
        graph,
        ir_version=8,
        opset_imports=[helper.make_opsetid("", 18)],
    )
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)


def equation_rank(equation: str) -> tuple[int, int, int, str]:
    operands = equation.split("->", 1)[0].split(",")
    contracted = set("".join(operands)) - set("nchw")
    diagonals = sum(len(operand) - len(set(operand)) for operand in operands)
    direct_sources = sum(operand == "nchw" for operand in operands)
    return (len(contracted), diagonals, -direct_sources, equation)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-known", action="store_true")
    parser.add_argument(
        "--family",
        choices=("mask", "general", "triple-mask", "triple-relayout", "triple-general"),
        default="mask",
    )
    parser.add_argument("--task-start", type=int, default=1)
    parser.add_argument("--task-end", type=int, default=400)
    parser.add_argument(
        "--tasks",
        help="Optional comma-separated task ids; overrides the numeric range.",
    )
    args = parser.parse_args()

    equations = {
        "mask": mask_equations,
        "general": general_equations,
        "triple-mask": triple_mask_equations,
        "triple-relayout": triple_relayout_equations,
        "triple-general": triple_general_equations,
    }[args.family]()
    hits: list[dict[str, str]] = []
    selected = None
    if args.tasks:
        selected = {
            item if item.startswith("task") else f"task{int(item):03d}"
            for item in args.tasks.split(",")
        }
    task_paths = [
        path
        for path in sorted(TASK_ROOT.glob("task*.json"))
        if (
            path.stem in selected
            if selected is not None
            else args.task_start <= int(path.stem[4:]) <= args.task_end
        )
    ]
    for task_path in task_paths:
        task = task_path.stem
        pairs = examples(task_path)
        matched_equations = []
        for equation in equations:
            if matches(equation, pairs):
                matched_equations.append(equation)
        if matched_equations and (
            args.include_known or task not in {"task067", "task179", "task241"}
        ):
            equation = min(matched_equations, key=equation_rank)
            candidate = args.output_dir / task / f"{equation.replace(',', '_').replace('->', '_')}.onnx"
            build(equation, candidate)
            hit = {
                "task": task,
                "equation": equation,
                "equivalent_equation_count": len(matched_equations),
                "candidate": str(candidate),
            }
            hits.append(hit)
            print(json.dumps(hit), flush=True)
    print(json.dumps({"equations": len(equations), "hits": hits}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
