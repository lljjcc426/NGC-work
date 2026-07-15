#!/usr/bin/env python
"""Build and exhaustively validate structural task092 candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


DEFAULT_BASELINE = Path(
    r"F:\kaggle\neurogolf-2026\work\e_high737801_recovered7_v1\models\task092.onnx"
)
DEFAULT_DATA = Path(r"F:\kaggle\neurogolf-2026\data\task092.json")
DEFAULT_OUTPUT_ROOT = Path(
    r"F:\kaggle\NGC-work\workplace E\optimized_onnx\task092_deep_20260713"
)
DEFAULT_REPORT = Path(
    r"F:\kaggle\NGC-work\workplace E\e_task092_deep_20260713_build.csv"
)


def initializer(name: str, values: object, dtype: np.dtype) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(values, dtype=dtype), name=name)


def node(op_type: str, inputs: list[str], output: str, **attrs: object) -> onnx.NodeProto:
    return helper.make_node(op_type, inputs, [output], name=output, **attrs)


def qlinear_matmul(a: str, b: str, output: str) -> onnx.NodeProto:
    return node(
        "QLinearMatMul",
        [a, "q_scale", "q_zero", b, "q_scale", "q_zero", "q_scale", "q_zero"],
        output,
    )


def build_rank12(output_path: Path, cross_guard: bool) -> None:
    """Render all line segments and the invalid-grid sentinel in one uint8 matmul."""
    nodes: list[onnx.NodeProto] = [
        node("ReduceMax", ["input"], "row_valid_f", axes=[1, 3], keepdims=1),
        node("ReduceMax", ["input"], "col_valid_f", axes=[1, 2], keepdims=1),
        node("Cast", ["row_valid_f"], "row_valid", to=TensorProto.BOOL),
        node("Cast", ["col_valid_f"], "col_valid", to=TensorProto.BOOL),
        node("Einsum", ["input", "coord_f"], "sum_x_f", equation="bchw,w->bc"),
        node("Einsum", ["input", "coord_f"], "sum_y_f", equation="bchw,h->bc"),
        node("Einsum", ["input", "coord_sq_f"], "sum_x2_f", equation="bchw,w->bc"),
        node("Einsum", ["input", "coord_sq_f"], "sum_y2_f", equation="bchw,h->bc"),
        node("Cast", ["sum_x_f"], "sum_x", to=TensorProto.FLOAT16),
        node("Cast", ["sum_y_f"], "sum_y", to=TensorProto.FLOAT16),
        node("Cast", ["sum_x2_f"], "sum_x2", to=TensorProto.FLOAT16),
        node("Cast", ["sum_y2_f"], "sum_y2", to=TensorProto.FLOAT16),
        node("Add", ["sum_x2", "sum_x2"], "twice_x2"),
        node("Mul", ["sum_x", "sum_x"], "square_x"),
        node("Sub", ["twice_x2", "square_x"], "delta_x2"),
        node("Sqrt", ["delta_x2"], "delta_x"),
        node("Add", ["sum_y2", "sum_y2"], "twice_y2"),
        node("Mul", ["sum_y", "sum_y"], "square_y"),
        node("Sub", ["twice_y2", "square_y"], "delta_y2"),
        node("Sqrt", ["delta_y2"], "delta_y"),
        node("Sub", ["sum_x", "delta_x"], "min_x_twice"),
        node("Mul", ["min_x_twice", "half_f16"], "min_x_f"),
        node("Sub", ["sum_x", "min_x_f"], "max_x_f"),
        node("Sub", ["sum_y", "delta_y"], "min_y_twice"),
        node("Mul", ["min_y_twice", "half_f16"], "min_y_f"),
        node("Sub", ["sum_y", "min_y_f"], "max_y_f"),
        node("Cast", ["min_x_f"], "min_x_all", to=TensorProto.UINT8),
        node("Cast", ["max_x_f"], "max_x_all", to=TensorProto.UINT8),
        node("Cast", ["min_y_f"], "min_y_all", to=TensorProto.UINT8),
        node("Cast", ["max_y_f"], "max_y_all", to=TensorProto.UINT8),
        node("Greater", ["delta_x", "zero_f16"], "horizontal_all"),
        node("Greater", ["delta_y", "zero_f16"], "vertical_all"),
        node("Or", ["horizontal_all", "vertical_all"], "active_all"),
        node("Cast", ["active_all"], "active_f16", to=TensorProto.FLOAT16),
        node("TopK", ["active_f16", "five_i64"], "active_values", axis=1, largest=1, sorted=0),
    ]
    # TopK has two outputs and is clearer when constructed explicitly.
    nodes[-1].output.extend(["active_indices"])
    nodes.extend(
        [
            node("Cast", ["active_indices"], "colors", to=TensorProto.UINT8),
            node("GatherElements", ["min_x_all", "active_indices"], "min_x", axis=1),
            node("GatherElements", ["max_x_all", "active_indices"], "max_x", axis=1),
            node("GatherElements", ["min_y_all", "active_indices"], "min_y", axis=1),
            node("GatherElements", ["max_y_all", "active_indices"], "max_y", axis=1),
        ]
    )

    # B-side slot tensors are [1, 1, 5, 1]; A-side tensors are [1, 1, 1, 5].
    for source in ("min_x", "max_x", "min_y", "max_y", "colors"):
        nodes.append(node("Unsqueeze", [source, "axes_b"], f"{source}_b"))
        nodes.append(node("Unsqueeze", [source, "axes_a"], f"{source}_a"))

    nodes.extend(
        [
            node("LessOrEqual", ["min_x_b", "x_coord"], "h_after_start"),
            node("LessOrEqual", ["x_coord", "max_x_b"], "h_before_end"),
            node("And", ["h_after_start", "h_before_end"], "h_between"),
            node("Less", ["min_x_b", "max_x_b"], "is_horizontal"),
            node("And", ["h_between", "is_horizontal"], "h_span"),
            node("LessOrEqual", ["min_y_a", "y_coord"], "v_after_start"),
            node("LessOrEqual", ["y_coord", "max_y_a"], "v_before_end"),
            node("And", ["v_after_start", "v_before_end"], "v_between"),
            node("Less", ["min_y_a", "max_y_a"], "is_vertical"),
            node("And", ["v_between", "is_vertical"], "v_span"),
            node("Equal", ["y_coord", "min_y_a"], "h_row"),
            node("Cast", ["h_row"], "h_row_u8", to=TensorProto.UINT8),
            node("Equal", ["x_coord", "min_x_b"], "v_col"),
            node("Cast", ["v_col"], "v_col_u8", to=TensorProto.UINT8),
        ]
    )

    h_mask = "h_span"
    if cross_guard:
        nodes.extend(
            [
                node("LessOrEqual", ["min_y_a", "min_y_b"], "cross_after_start"),
                node("LessOrEqual", ["min_y_b", "max_y_a"], "cross_before_end"),
                node("And", ["cross_after_start", "cross_before_end"], "cross_between"),
                node("And", ["cross_between", "is_vertical"], "cross_pairs"),
                node("Cast", ["cross_pairs"], "cross_pairs_u8", to=TensorProto.UINT8),
                qlinear_matmul("cross_pairs_u8", "v_col_u8", "cross_columns"),
                node("Equal", ["cross_columns", "zero_u8"], "no_vertical_cross"),
                node("And", ["h_span", "no_vertical_cross"], "h_span_guarded"),
            ]
        )
        h_mask = "h_span_guarded"

    nodes.extend(
        [
            node("Where", [h_mask, "colors_b", "zero_u8"], "h_span_color"),
            node("Where", ["v_span", "colors_a", "zero_u8"], "v_span_color"),
            node("Not", ["row_valid"], "row_invalid"),
            node("Cast", ["row_invalid"], "row_invalid_u8", to=TensorProto.UINT8),
            node("Cast", ["row_valid"], "row_valid_u8", to=TensorProto.UINT8),
            node("Where", ["col_valid", "zero_u8", "ten_u8"], "col_invalid_code"),
            node(
                "Concat",
                ["h_row_u8", "v_span_color", "row_invalid_u8", "row_valid_u8"],
                "rank_a",
                axis=3,
            ),
            node(
                "Concat",
                ["h_span_color", "v_col_u8", "outside_row_code", "col_invalid_code"],
                "rank_b",
                axis=2,
            ),
            qlinear_matmul("rank_a", "rank_b", "output_codes"),
            node("Equal", ["output_codes", "channel_codes"], "output"),
        ]
    )

    initializers = [
        initializer("coord_f", np.arange(30), np.float32),
        initializer("coord_sq_f", np.arange(30) ** 2, np.float32),
        initializer("x_coord", np.arange(30).reshape(1, 1, 1, 30), np.uint8),
        initializer("y_coord", np.arange(30).reshape(1, 1, 30, 1), np.uint8),
        initializer("channel_codes", np.arange(10).reshape(1, 10, 1, 1), np.uint8),
        initializer("outside_row_code", np.full((1, 1, 1, 30), 10), np.uint8),
        initializer("half_f16", 0.5, np.float16),
        initializer("zero_f16", 0.0, np.float16),
        initializer("zero_u8", 0, np.uint8),
        initializer("ten_u8", 10, np.uint8),
        initializer("five_i64", [5], np.int64),
        initializer("axes_b", [1, 3], np.int64),
        initializer("axes_a", [1, 2], np.int64),
        initializer("q_scale", 1.0, np.float32),
        initializer("q_zero", 0, np.uint8),
    ]
    graph = helper.make_graph(
        nodes,
        f"task092_rank12_qmm_cross_guard_{int(cross_guard)}",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])],
        initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 16)],
        ir_version=8,
        producer_name="ngc-workplace-e",
    )
    onnx.checker.check_model(model, full_check=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)


def encode(grid: list[list[int]]) -> np.ndarray:
    tensor = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for row, cells in enumerate(grid):
        for col, color in enumerate(cells):
            tensor[0, color, row, col] = 1.0
    return tensor


def axis_fill_transposed(grid: list[list[int]]) -> list[list[int]]:
    return [
        [cell or sum(set(line[:index]) & set(line[index:])) for index, cell in enumerate(line)]
        for line in zip(*grid)
    ]


def reference_transform(grid: list[list[int]]) -> list[list[int]]:
    return axis_fill_transposed(axis_fill_transposed(grid))


def load_examples(data_path: Path) -> dict[str, list[dict[str, list[list[int]]]]]:
    return json.loads(data_path.read_text(encoding="utf-8"))


def verify_reference(examples: dict[str, list[dict[str, list[list[int]]]]]) -> int:
    checked = 0
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(examples[split]):
            actual = reference_transform(example["input"])
            if actual != example["output"]:
                raise RuntimeError(f"reference mismatch at {split}[{index}]")
            checked += 1
    return checked


def verify_model(
    model_path: Path,
    examples: dict[str, list[dict[str, list[list[int]]]]],
) -> tuple[int, int, dict[str, tuple[int, int]]]:
    session = ort.InferenceSession(model_path.read_bytes(), providers=["CPUExecutionProvider"])
    passed = 0
    failed = 0
    splits: dict[str, tuple[int, int]] = {}
    for split in ("train", "test", "arc-gen"):
        split_passed = 0
        split_failed = 0
        for example in examples[split]:
            actual = session.run(["output"], {"input": encode(example["input"])})[0]
            expected = encode(example["output"]) > 0
            if np.array_equal(actual > 0, expected):
                split_passed += 1
            else:
                split_failed += 1
        splits[split] = (split_passed, split_failed)
        passed += split_passed
        failed += split_failed
    return passed, failed, splits


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    if not args.baseline.is_file():
        raise FileNotFoundError(args.baseline)
    examples = load_examples(args.data)
    reference_checked = verify_reference(examples)
    candidates = {
        "rank12_qmm_no_cross_guard_rejected": False,
        "rank12_qmm_cross_guard": True,
    }
    rows: list[dict[str, object]] = []
    for name, cross_guard in candidates.items():
        model_path = args.output_root / name / "task092.onnx"
        build_rank12(model_path, cross_guard=cross_guard)
        passed, failed, split_counts = verify_model(model_path, examples)
        rows.append(
            {
                "candidate": name,
                "path": str(model_path),
                "reference_examples": reference_checked,
                "train_pass": split_counts["train"][0],
                "train_fail": split_counts["train"][1],
                "test_pass": split_counts["test"][0],
                "test_fail": split_counts["test"][1],
                "arc_gen_pass": split_counts["arc-gen"][0],
                "arc_gen_fail": split_counts["arc-gen"][1],
                "total_pass": passed,
                "total_fail": failed,
                "filesize": model_path.stat().st_size,
                "sha256": sha256(model_path),
            }
        )
        print(f"{name}: {passed}/{passed + failed}, sha256={rows[-1]['sha256']}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return 0 if rows[-1]["total_fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
