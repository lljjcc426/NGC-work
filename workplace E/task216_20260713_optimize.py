from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
from onnx import TensorProto, helper, numpy_helper


TASK = 216
HERE = Path(__file__).resolve().parent
NGC_ROOT = Path(r"F:\kaggle\neurogolf-2026")
DEFAULT_SOURCE = (
    NGC_ROOT / "work" / "e_high737801_recovered7_v1" / "models" / "task216.onnx"
)
DEFAULT_OUTPUT = (
    HERE
    / "optimized_onnx"
    / "task216_20260713_compact_coords"
    / "task216.onnx"
)
DEFAULT_CSV = HERE / "task216_20260713_candidates.csv"
DATA_PATH = NGC_ROOT / "data" / "task216.json"


def tensor(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(value, name=name)


def build_model() -> onnx.ModelProto:
    initializers = [
        tensor("x_scale", np.asarray(1.0, dtype=np.float32)),
        tensor("corner_y_scale", np.asarray(5.0, dtype=np.float32)),
        tensor("x_zp", np.asarray(0, dtype=np.uint8)),
        tensor("w_zp", np.asarray(1, dtype=np.uint8)),
        tensor(
            "corner_w",
            np.asarray([[[[1, 0], [0, 4]], [[1, 0], [0, 4]]]], dtype=np.uint8),
        ),
        tensor("slice_starts", np.asarray([0, 1, 0, 0], dtype=np.int64)),
        tensor("slice_ends", np.asarray([1, 3, 20, 20], dtype=np.int64)),
        tensor("twenty", np.asarray([20], dtype=np.int32)),
        tensor("eighteen", np.asarray([18], dtype=np.int32)),
        tensor("slot_y_scale", np.asarray(3.0, dtype=np.float32)),
        tensor(
            "slot_w",
            np.concatenate(
                [
                    np.full((1, 1, 1, 17), 2, dtype=np.uint8),
                    np.ones((1, 1, 1, 17), dtype=np.uint8),
                ],
                axis=0,
            ),
        ),
        tensor("coords_row", np.arange(20, dtype=np.int32)[None, :]),
        tensor("cols_shape", np.asarray([4, 1], dtype=np.int64)),
        tensor("one_i64", np.asarray([1], dtype=np.int64)),
        tensor(
            "count_diag_idx",
            np.asarray(
                [[[1, 0, 0], [1, 1, 1], [1, 2, 2], [1, 3, 3]]],
                dtype=np.int64,
            ),
        ),
        tensor("starts_prefix32", np.asarray([0, 0], dtype=np.int32)),
        tensor("ends_prefix32", np.asarray([1, 2], dtype=np.int32)),
        tensor("pad_prefix", np.asarray([0, 1, 0, 0, 0, 7], dtype=np.int64)),
        tensor("thirty", np.asarray(30, dtype=np.int32)),
    ]

    nodes = [
        helper.make_node(
            "Slice", ["input", "slice_starts", "slice_ends"], ["c12_f"]
        ),
        helper.make_node("Cast", ["c12_f"], ["c12"], to=TensorProto.UINT8),
        helper.make_node(
            "QLinearConv",
            [
                "c12",
                "x_scale",
                "x_zp",
                "corner_w",
                "x_scale",
                "w_zp",
                "corner_y_scale",
                "x_zp",
            ],
            ["cmap"],
            kernel_shape=[2, 2],
            pads=[1, 1, -2, -3],
        ),
        helper.make_node(
            "QLinearConv",
            [
                "cmap",
                "x_scale",
                "x_zp",
                "slot_w",
                "x_scale",
                "x_zp",
                "slot_y_scale",
                "x_zp",
            ],
            ["row_slots"],
            kernel_shape=[1, 17],
        ),
        helper.make_node("Flatten", ["row_slots"], ["slots_flat"], axis=1),
        helper.make_node("ArgMax", ["slots_flat"], ["a0"], axis=1, keepdims=0),
        helper.make_node(
            "ArgMax",
            ["slots_flat"],
            ["a1"],
            axis=1,
            keepdims=0,
            select_last_index=1,
        ),
        helper.make_node("Add", ["a0", "one_i64"], ["mid_start"]),
        helper.make_node(
            "Slice",
            ["slots_flat", "mid_start", "a1", "one_i64"],
            ["slots_mid"],
        ),
        helper.make_node(
            "ArgMax", ["slots_mid"], ["a2_rel"], axis=1, keepdims=0
        ),
        helper.make_node(
            "ArgMax",
            ["slots_mid"],
            ["a3_rel"],
            axis=1,
            keepdims=0,
            select_last_index=1,
        ),
        helper.make_node("Cast", ["a0"], ["a0_i32"], to=TensorProto.INT32),
        helper.make_node("Cast", ["a1"], ["a1_i32"], to=TensorProto.INT32),
        helper.make_node(
            "Cast", ["mid_start"], ["mid_start_i32"], to=TensorProto.INT32
        ),
        helper.make_node(
            "Cast", ["a2_rel"], ["a2_rel_i32"], to=TensorProto.INT32
        ),
        helper.make_node(
            "Cast", ["a3_rel"], ["a3_rel_i32"], to=TensorProto.INT32
        ),
        helper.make_node("Add", ["mid_start_i32", "a2_rel_i32"], ["a2_i32"]),
        helper.make_node("Add", ["mid_start_i32", "a3_rel_i32"], ["a3_i32"]),
        helper.make_node(
            "Concat",
            ["a0_i32", "a1_i32", "a2_i32", "a3_i32"],
            ["idx"],
            axis=0,
        ),
        helper.make_node("Mod", ["idx", "eighteen"], ["rows"], fmod=0),
        helper.make_node(
            "GreaterOrEqual", ["idx", "eighteen"], ["second_row_slot"]
        ),
        helper.make_node("Gather", ["cmap", "rows"], ["corner_rows"], axis=2),
        helper.make_node(
            "ArgMax", ["corner_rows"], ["first_cols64"], axis=3, keepdims=0
        ),
        helper.make_node(
            "ArgMax",
            ["corner_rows"],
            ["last_cols64"],
            axis=3,
            keepdims=0,
            select_last_index=1,
        ),
        helper.make_node(
            "Cast", ["first_cols64"], ["first_cols"], to=TensorProto.INT32
        ),
        helper.make_node(
            "Cast", ["last_cols64"], ["last_cols"], to=TensorProto.INT32
        ),
        helper.make_node(
            "Where", ["second_row_slot", "last_cols", "first_cols"], ["cols_3d"]
        ),
        helper.make_node("Reshape", ["cols_3d", "cols_shape"], ["cols"]),
        helper.make_node("Gather", ["c12", "rows"], ["row_cands"], axis=2),
        helper.make_node(
            "ReduceMax", ["row_cands"], ["row_vecs"], axes=[0, 1], keepdims=0
        ),
        helper.make_node("Less", ["coords_row", "cols"], ["before_w_b"]),
        helper.make_node("Where", ["before_w_b", "w_zp", "row_vecs"], ["maskw"]),
        helper.make_node(
            "ArgMin", ["maskw"], ["wend64"], axis=1, keepdims=1
        ),
        helper.make_node("Cast", ["wend64"], ["wend"], to=TensorProto.INT32),
        helper.make_node("LessOrEqual", ["wend", "cols"], ["no_wend"]),
        helper.make_node("Where", ["no_wend", "twenty", "wend"], ["colend"]),
        helper.make_node("Gather", ["c12", "cols"], ["col_cands"], axis=3),
        helper.make_node(
            "ReduceMax",
            ["col_cands"],
            ["col_vecs"],
            axes=[0, 1, 4],
            keepdims=0,
        ),
        helper.make_node("Transpose", ["col_vecs"], ["col_vecs_t"], perm=[1, 0]),
        helper.make_node("Unsqueeze", ["rows"], ["rows_col"], axes=[1]),
        helper.make_node("Less", ["coords_row", "rows_col"], ["before_h_b"]),
        helper.make_node(
            "Where", ["before_h_b", "w_zp", "col_vecs_t"], ["maskh"]
        ),
        helper.make_node(
            "ArgMin", ["maskh"], ["hend64"], axis=1, keepdims=1
        ),
        helper.make_node("Cast", ["hend64"], ["hend"], to=TensorProto.INT32),
        helper.make_node("LessOrEqual", ["hend", "rows_col"], ["no_hend"]),
        helper.make_node("Where", ["no_hend", "twenty", "hend"], ["rowend"]),
        helper.make_node(
            "Less", ["coords_row", "rowend"], ["before_rowend_b"]
        ),
        helper.make_node(
            "Xor", ["before_h_b", "before_rowend_b"], ["row_mask_b"]
        ),
        helper.make_node(
            "Cast", ["row_mask_b"], ["row_mask"], to=TensorProto.UINT8
        ),
        helper.make_node(
            "Less", ["coords_row", "colend"], ["before_colend_b"]
        ),
        helper.make_node(
            "Xor", ["before_w_b", "before_colend_b"], ["col_mask_b"]
        ),
        helper.make_node(
            "Cast", ["col_mask_b"], ["col_mask"], to=TensorProto.UINT8
        ),
        helper.make_node(
            "Transpose", ["col_mask"], ["col_mask_t"], perm=[1, 0]
        ),
        helper.make_node(
            "QLinearMatMul",
            [
                "row_mask",
                "x_scale",
                "x_zp",
                "c12",
                "x_scale",
                "x_zp",
                "x_scale",
                "x_zp",
            ],
            ["row_counts"],
        ),
        helper.make_node(
            "QLinearMatMul",
            [
                "row_counts",
                "x_scale",
                "x_zp",
                "col_mask_t",
                "x_scale",
                "x_zp",
                "x_scale",
                "x_zp",
            ],
            ["color_counts"],
        ),
        helper.make_node(
            "GatherND",
            ["color_counts", "count_diag_idx"],
            ["counts"],
            batch_dims=1,
        ),
        helper.make_node("ArgMax", ["counts"], ["winner"], axis=1, keepdims=0),
        helper.make_node("Gather", ["rows", "winner"], ["w_row"], axis=0),
        helper.make_node("Gather", ["cols", "winner"], ["w_col_2d"], axis=0),
        helper.make_node("Squeeze", ["w_col_2d"], ["w_col"], axes=[1]),
        helper.make_node(
            "Gather", ["rowend", "winner"], ["w_rowend_2d"], axis=0
        ),
        helper.make_node("Squeeze", ["w_rowend_2d"], ["w_rowend"], axes=[1]),
        helper.make_node(
            "Gather", ["colend", "winner"], ["w_colend_2d"], axis=0
        ),
        helper.make_node("Squeeze", ["w_colend_2d"], ["w_colend"], axes=[1]),
        helper.make_node(
            "Concat",
            ["starts_prefix32", "w_row", "w_col"],
            ["crop_starts"],
            axis=0,
        ),
        helper.make_node(
            "Concat",
            ["ends_prefix32", "w_rowend", "w_colend"],
            ["crop_ends"],
            axis=0,
        ),
        helper.make_node("Slice", ["c12", "crop_starts", "crop_ends"], ["crop"]),
        helper.make_node("Sub", ["w_rowend", "w_row"], ["hh"]),
        helper.make_node("Sub", ["w_colend", "w_col"], ["ww"]),
        helper.make_node("Sub", ["thirty", "hh"], ["padh"]),
        helper.make_node("Sub", ["thirty", "ww"], ["padw"]),
        helper.make_node("Cast", ["padh"], ["padh1"], to=TensorProto.INT64),
        helper.make_node("Cast", ["padw"], ["padw1"], to=TensorProto.INT64),
        helper.make_node(
            "Concat", ["pad_prefix", "padh1", "padw1"], ["pads"], axis=0
        ),
        helper.make_node("Pad", ["crop", "pads"], ["output"], mode="constant"),
    ]

    graph = helper.make_graph(
        nodes,
        "task216_20260713_compact_coords",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.UINT8, [1, 10, 30, 30])],
        initializers,
        value_info=[
            helper.make_tensor_value_info("crop", TensorProto.UINT8, [1, 2, 1, 1]),
            helper.make_tensor_value_info("slots_mid", TensorProto.UINT8, [1, 1]),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="ngc_workplace_e_task216_20260713",
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 12)],
    )
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model


def encode(grid: list[list[int]]) -> np.ndarray:
    result = np.zeros((1, 10, 30, 30), dtype=np.float32)
    values = np.asarray(grid)
    for color in range(10):
        result[0, color, : values.shape[0], : values.shape[1]] = values == color
    return result


def expected(grid: list[list[int]]) -> np.ndarray:
    return encode(grid)


def verify_all(model: onnx.ModelProto) -> dict[str, tuple[int, int]]:
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = onnxruntime.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    examples = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    results: dict[str, tuple[int, int]] = {}
    for split in ("train", "test", "arc-gen"):
        passed = 0
        for example in examples[split]:
            actual = session.run(["output"], {"input": encode(example["input"])})[0]
            passed += int(np.array_equal((actual > 0).astype(np.float32), expected(example["output"])))
        results[split] = (passed, len(examples[split]))
    return results


def parameter_count(model: onnx.ModelProto) -> int:
    return sum(int(np.prod(item.dims)) for item in model.graph.initializer)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    source = onnx.load(args.source)
    candidate = build_model()
    validation = verify_all(candidate)
    if any(passed != total for passed, total in validation.values()):
        raise RuntimeError(f"candidate failed full validation: {validation}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(candidate, args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    row = {
        "task": "task216",
        "variant": "binary_corner_slot_conv_compact_coords",
        "source": str(args.source),
        "candidate": str(args.output),
        "source_nodes": len(source.graph.node),
        "candidate_nodes": len(candidate.graph.node),
        "source_params": parameter_count(source),
        "candidate_params": parameter_count(candidate),
        "filesize": args.output.stat().st_size,
        "train": f"{validation['train'][0]}/{validation['train'][1]}",
        "test": f"{validation['test'][0]}/{validation['test'][1]}",
        "arc_gen": f"{validation['arc-gen'][0]}/{validation['arc-gen'][1]}",
        "verified_all": True,
        "sha256": digest,
    }
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(json.dumps(row, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
