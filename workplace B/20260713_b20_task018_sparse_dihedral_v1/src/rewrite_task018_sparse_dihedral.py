from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "reconstruction_candidates" / "b_task018_sparse_dihedral_v1" / "task018.onnx"


class GraphBuilder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self._counter = 0

    def init(self, name: str, value: np.ndarray) -> str:
        self.initializers.append(numpy_helper.from_array(value, name=name))
        return name

    def node(
        self,
        op_type: str,
        inputs: list[str],
        prefix: str,
        outputs: int = 1,
        **attrs: object,
    ) -> str | tuple[str, ...]:
        index = self._counter
        self._counter += 1
        names = tuple(f"{prefix}_{index}_{part}" for part in range(outputs))
        self.nodes.append(
            helper.make_node(
                op_type,
                inputs,
                list(names),
                name=f"{prefix}_{index}",
                **attrs,
            )
        )
        return names[0] if outputs == 1 else names


def _indices(builder: GraphBuilder, mask: str, prefix: str) -> str:
    nonzero = builder.node("NonZero", [mask], f"{prefix}_nonzero")
    return builder.node("Squeeze", [nonzero, "axes_zero"], f"{prefix}_indices")


def _filter(builder: GraphBuilder, values: str, mask: str, prefix: str) -> str:
    indices = _indices(builder, mask, prefix)
    return builder.node("Gather", [values, indices], f"{prefix}_gather", axis=0)


def _select_two(
    builder: GraphBuilder,
    coords: str,
    mask: str,
    prefix: str,
) -> tuple[str, str]:
    mask_f16 = builder.node("Cast", [mask], f"{prefix}_mask_f16", to=TensorProto.FLOAT16)
    scores = builder.node("Mul", [mask_f16, "rank_30"], f"{prefix}_scores")
    values, indices = builder.node(
        "TopK",
        [scores, "top2"],
        f"{prefix}_top2",
        outputs=2,
        axis=0,
    )
    selected = builder.node("Gather", [coords, indices], f"{prefix}_coords", axis=0)
    valid = builder.node("Greater", [values, "zero_f16"], f"{prefix}_valid")
    return selected, valid


def _nearest_indices(builder: GraphBuilder, left: str, right: str, prefix: str) -> str:
    left_u = builder.node("Unsqueeze", [left, "axes_one"], f"{prefix}_left")
    right_u = builder.node("Unsqueeze", [right, "axes_zero"], f"{prefix}_right")
    delta = builder.node("Sub", [left_u, right_u], f"{prefix}_delta")
    distance = builder.node("Abs", [delta], f"{prefix}_abs")
    distance_sum = builder.node(
        "ReduceSum",
        [distance, "axes_coord"],
        f"{prefix}_distance",
        keepdims=0,
    )
    return builder.node("ArgMin", [distance_sum], f"{prefix}_argmin", axis=1, keepdims=0)


def build_model() -> onnx.ModelProto:
    b = GraphBuilder()

    b.init("label_kernel", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))
    b.init("color_ids_u8", np.arange(10, dtype=np.uint8).reshape(1, 10))
    b.init("nonbackground", np.array([[False] + [True] * 9], dtype=np.bool_))
    b.init("nonbackground_f", np.array([[0.0] + [1.0] * 9], dtype=np.float32))
    b.init("color_tie", np.arange(16, 6, -1, dtype=np.float32).reshape(1, 10))
    b.init("channels_u8", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1))
    b.init(
        "dihedral_matrices",
        np.array(
            [
                [[1, 0], [0, 1]],
                [[1, 0], [0, -1]],
                [[-1, 0], [0, 1]],
                [[-1, 0], [0, -1]],
                [[0, 1], [1, 0]],
                [[0, -1], [1, 0]],
                [[0, 1], [-1, 0]],
                [[0, -1], [-1, 0]],
            ],
            dtype=np.float16,
        ),
    )
    b.init("axes_hw", np.array([2, 3], dtype=np.int64))
    b.init("axes_row", np.array([1, 3], dtype=np.int64))
    b.init("axes_col", np.array([1, 2], dtype=np.int64))
    b.init("axes_coord", np.array([2], dtype=np.int64))
    b.init("axes_zero", np.array([0], dtype=np.int64))
    b.init("axes_zero_one", np.array([0, 1], dtype=np.int64))
    b.init("axes_zero_two", np.array([0, 2], dtype=np.int64))
    b.init("axes_one", np.array([1], dtype=np.int64))
    b.init("axes_one_two", np.array([1, 2], dtype=np.int64))
    b.init("axes_two", np.array([2], dtype=np.int64))
    b.init("axes_last", np.array([-1], dtype=np.int64))
    b.init("slice_zero", np.array([0], dtype=np.int64))
    b.init("slice_two", np.array([2], dtype=np.int64))
    b.init("slice_axis_one", np.array([1], dtype=np.int64))
    b.init("shape_24", np.array([24, 24], dtype=np.int64))
    b.init("shape_576", np.array([576], dtype=np.int64))
    b.init("shape_flat_candidates", np.array([0, -1], dtype=np.int64))
    b.init("shape_coords", np.array([-1, 2], dtype=np.int64))
    b.init("shape_values", np.array([-1], dtype=np.int64))
    b.init("top3", np.array([3], dtype=np.int64))
    b.init("top30", np.array([30], dtype=np.int64))
    b.init("top2", np.array([2], dtype=np.int64))
    b.init("rank_30", np.arange(30, 0, -1, dtype=np.float16))
    b.init("idx0", np.array([0], dtype=np.int64))
    b.init("idx1", np.array([1], dtype=np.int64))
    b.init("idx2", np.array([2], dtype=np.int64))
    b.init("coord0", np.array(0, dtype=np.int64))
    b.init("coord1", np.array(1, dtype=np.int64))
    b.init("zero_u8", np.array(0, dtype=np.uint8))
    b.init("sentinel_u8", np.array(255, dtype=np.uint8))
    b.init("zero_f32", np.array(0, dtype=np.float32))
    b.init("zero_f16", np.array(0, dtype=np.float16))
    b.init("limit_f16", np.array(24, dtype=np.float16))
    b.init("max_index_f16", np.array(23, dtype=np.float16))
    b.init("far_coord_f16", np.array([1000, 1000], dtype=np.float16))
    b.init("eight_i64", np.array(8, dtype=np.int64))
    b.init("one_i64", np.array([1, 1], dtype=np.int64))
    b.init("thirty_i64", np.array([30, 30], dtype=np.int64))
    b.init("zero_pads", np.array([0, 0], dtype=np.int64))
    b.init("slice_axes_2d", np.array([0, 1], dtype=np.int64))
    b.init("slice_24", np.array([24, 24], dtype=np.int64))
    b.init("pads_24_to_30", np.array([0, 0, 6, 6], dtype=np.int64))
    b.init("row_ids_24", np.arange(24, dtype=np.int64).reshape(24, 1))
    b.init("col_ids_24", np.arange(24, dtype=np.int64).reshape(1, 24))
    b.init("dummy_coord_f16", np.array([0, 0], dtype=np.float16))
    b.init("empty_u8_value", np.array([0], dtype=np.uint8))

    label_f32 = b.node(
        "Conv",
        ["input", "label_kernel"],
        "label",
        pads=[0, 0, -6, -6],
    )
    label_u8_4d = b.node("Cast", [label_f32], "label_u8", to=TensorProto.UINT8)

    counts = b.node("ReduceSum", ["input", "axes_hw"], "color_counts", keepdims=0)
    foreground_counts = b.node("Mul", [counts, "nonbackground_f"], "foreground_counts")
    dominant_i64 = b.node("ArgMax", [foreground_counts], "dominant_i64", axis=1, keepdims=1)
    dominant_u8 = b.node("Cast", [dominant_i64], "dominant_u8", to=TensorProto.UINT8)
    positive = b.node("Greater", [counts, "zero_f32"], "positive_colors")
    foreground = b.node("And", [positive, "nonbackground"], "foreground_colors")
    dominant_color = b.node("Equal", ["color_ids_u8", dominant_u8], "dominant_color")
    not_dominant = b.node("Not", [dominant_color], "not_dominant_color")
    candidates = b.node("And", [foreground, not_dominant], "marker_colors")
    candidate_f = b.node("Cast", [candidates], "candidate_f", to=TensorProto.FLOAT)
    ranked = b.node("Mul", [candidate_f, "color_tie"], "ranked_colors")
    _, marker_ids = b.node("TopK", [ranked, "top3"], "marker_top3", outputs=2, axis=1)
    anchor_raw = b.node("Gather", [marker_ids, "idx0"], "anchor_color_raw", axis=1)
    marker_c_raw = b.node("Gather", [marker_ids, "idx1"], "marker_c_raw", axis=1)
    marker_b_raw = b.node("Gather", [marker_ids, "idx2"], "marker_b_raw", axis=1)
    anchor_i64 = b.node("Squeeze", [anchor_raw, "axes_zero_one"], "anchor_color_i64")
    marker_c_i64 = b.node("Squeeze", [marker_c_raw, "axes_zero_one"], "marker_c_i64")
    marker_b_i64 = b.node("Squeeze", [marker_b_raw, "axes_zero_one"], "marker_b_i64")
    anchor_u8 = b.node("Cast", [anchor_i64], "anchor_color", to=TensorProto.UINT8)
    marker_b_u8 = b.node("Cast", [marker_b_i64], "marker_b", to=TensorProto.UINT8)
    marker_c_u8 = b.node("Cast", [marker_c_i64], "marker_c", to=TensorProto.UINT8)

    dominant_mask = b.node("Equal", [label_u8_4d, dominant_u8], "dominant_mask")
    dominant_mask_u8 = b.node("Cast", [dominant_mask], "dominant_mask_u8", to=TensorProto.UINT8)
    source_region_4d = b.node(
        "MaxPool",
        [dominant_mask_u8],
        "source_region",
        kernel_shape=[7, 7],
        pads=[3, 3, 3, 3],
        strides=[1, 1],
    )
    label_flat_u8 = b.node("Reshape", [label_u8_4d, "shape_576"], "label_flat_u8")
    label_flat_f16 = b.node("Cast", [label_flat_u8], "label_flat_f16", to=TensorProto.FLOAT16)
    all_values_f16, all_indices = b.node(
        "TopK",
        [label_flat_f16, "top30"],
        "colored_top30",
        outputs=2,
        axis=0,
    )
    all_colors = b.node("Cast", [all_values_f16], "colored_values", to=TensorProto.UINT8)
    all_indices_f16 = b.node("Cast", [all_indices], "colored_indices_f16", to=TensorProto.FLOAT16)
    all_rows_raw = b.node("Div", [all_indices_f16, "limit_f16"], "colored_rows_raw")
    all_rows = b.node("Floor", [all_rows_raw], "colored_rows")
    all_cols = b.node("Mod", [all_indices_f16, "limit_f16"], "colored_cols", fmod=1)
    all_rows_col = b.node("Unsqueeze", [all_rows, "axes_one"], "colored_rows_col")
    all_cols_col = b.node("Unsqueeze", [all_cols, "axes_one"], "colored_cols_col")
    all_coords = b.node("Concat", [all_rows_col, all_cols_col], "colored_coords", axis=1)
    colored_valid = b.node("Greater", [all_values_f16, "zero_f16"], "colored_valid")
    source_region_flat = b.node("Reshape", [source_region_4d, "shape_576"], "source_region_flat")
    region_values = b.node("Gather", [source_region_flat, all_indices], "colored_regions", axis=0)
    in_source_region = b.node("Greater", [region_values, "zero_u8"], "in_source_region")
    is_source = b.node("And", [colored_valid, in_source_region], "is_source")
    outside_source = b.node("Not", [in_source_region], "outside_source")
    is_target = b.node("And", [colored_valid, outside_source], "is_target")

    source_coords = all_coords
    source_colors = all_colors
    source_a_color = b.node("Equal", [source_colors, anchor_u8], "source_a_color")
    source_b_color = b.node("Equal", [source_colors, marker_b_u8], "source_b_color")
    source_c_color = b.node("Equal", [source_colors, marker_c_u8], "source_c_color")
    source_a_mask = b.node("And", [source_a_color, is_source], "source_a_mask")
    source_b_mask = b.node("And", [source_b_color, is_source], "source_b_mask")
    source_c_mask = b.node("And", [source_c_color, is_source], "source_c_mask")
    source_a_raw, source_a_valid = _select_two(b, source_coords, source_a_mask, "source_a")
    source_b_selected, source_b_valid = _select_two(b, source_coords, source_b_mask, "source_b")
    source_c_selected, source_c_valid = _select_two(b, source_coords, source_c_mask, "source_c")
    source_a_valid_xy = b.node("Unsqueeze", [source_a_valid, "axes_one"], "source_a_valid_xy")
    source_b_valid_xy = b.node("Unsqueeze", [source_b_valid, "axes_one"], "source_b_valid_xy")
    source_c_valid_xy = b.node("Unsqueeze", [source_c_valid, "axes_one"], "source_c_valid_xy")
    source_a = b.node("Where", [source_a_valid_xy, source_a_raw, "far_coord_f16"], "source_a_safe")
    source_b_raw = b.node(
        "Where",
        [source_b_valid_xy, source_b_selected, "far_coord_f16"],
        "source_b_safe",
    )
    source_c_raw = b.node(
        "Where",
        [source_c_valid_xy, source_c_selected, "far_coord_f16"],
        "source_c_safe",
    )

    target_a_color = b.node("Equal", [all_colors, anchor_u8], "target_a_color")
    target_a_mask = b.node("And", [target_a_color, is_target], "target_a_mask")
    target_a, target_a_valid = _select_two(b, all_coords, target_a_mask, "target_a")

    b_order = _nearest_indices(b, source_a, source_b_raw, "align_b")
    c_order = _nearest_indices(b, source_a, source_c_raw, "align_c")
    source_b = b.node("Gather", [source_b_raw, b_order], "source_b_aligned", axis=0)
    source_c = b.node("Gather", [source_c_raw, c_order], "source_c_aligned", axis=0)
    rel_b = b.node("Sub", [source_b, source_a], "relative_b")
    rel_c = b.node("Sub", [source_c, source_a], "relative_c")
    transformed_b = b.node(
        "Einsum",
        [rel_b, "dihedral_matrices"],
        "transformed_b",
        equation="si,tij->stj",
    )
    transformed_c = b.node(
        "Einsum",
        [rel_c, "dihedral_matrices"],
        "transformed_c",
        equation="si,tij->stj",
    )
    target_expanded = b.node("Unsqueeze", [target_a, "axes_one_two"], "target_expanded")
    transformed_b_expanded = b.node("Unsqueeze", [transformed_b, "axes_zero"], "b_expanded")
    transformed_c_expanded = b.node("Unsqueeze", [transformed_c, "axes_zero"], "c_expanded")
    candidate_b = b.node("Add", [target_expanded, transformed_b_expanded], "candidate_b")
    candidate_c = b.node("Add", [target_expanded, transformed_c_expanded], "candidate_c")

    def candidate_match(coords: str, color: str, prefix: str) -> str:
        ge_zero = b.node("GreaterOrEqual", [coords, "zero_f16"], f"{prefix}_ge_zero")
        lt_limit = b.node("Less", [coords, "limit_f16"], f"{prefix}_lt_limit")
        in_bounds = b.node("And", [ge_zero, lt_limit], f"{prefix}_in_bounds_xy")
        in_bounds_u8 = b.node("Cast", [in_bounds], f"{prefix}_in_bounds_u8", to=TensorProto.UINT8)
        all_in_bounds_u8 = b.node(
            "ReduceMin",
            [in_bounds_u8],
            f"{prefix}_in_bounds",
            axes=[3],
            keepdims=0,
        )
        all_in_bounds = b.node("Cast", [all_in_bounds_u8], f"{prefix}_in_bounds_b", to=TensorProto.BOOL)
        clipped = b.node("Clip", [coords, "zero_f16", "max_index_f16"], f"{prefix}_clipped")
        rows = b.node("Gather", [clipped, "coord0"], f"{prefix}_rows", axis=3)
        cols = b.node("Gather", [clipped, "coord1"], f"{prefix}_cols", axis=3)
        row_offsets = b.node("Mul", [rows, "limit_f16"], f"{prefix}_row_offsets")
        flat_f16 = b.node("Add", [row_offsets, cols], f"{prefix}_flat_f16")
        indices = b.node("Cast", [flat_f16], f"{prefix}_indices", to=TensorProto.INT64)
        values = b.node("Gather", [label_flat_u8, indices], f"{prefix}_values", axis=0)
        same_color = b.node("Equal", [values, color], f"{prefix}_same_color")
        return b.node("And", [all_in_bounds, same_color], f"{prefix}_match")

    b_match = candidate_match(candidate_b, marker_b_u8, "match_b")
    c_match = candidate_match(candidate_c, marker_c_u8, "match_c")
    marker_match = b.node("And", [b_match, c_match], "marker_match")
    source_valid_ab = b.node("And", [source_a_valid, source_b_valid], "source_valid_ab")
    source_valid = b.node("And", [source_valid_ab, source_c_valid], "source_valid")
    source_valid_expanded = b.node(
        "Unsqueeze",
        [source_valid, "axes_zero_two"],
        "source_valid_expanded",
    )
    target_valid_expanded = b.node(
        "Unsqueeze",
        [target_a_valid, "axes_one_two"],
        "target_valid_expanded",
    )
    active_candidates = b.node("And", [source_valid_expanded, target_valid_expanded], "active_candidates")
    valid_candidates = b.node("And", [marker_match, active_candidates], "valid_candidates")
    valid_u8 = b.node("Cast", [valid_candidates], "valid_candidates_u8", to=TensorProto.UINT8)
    valid_flat = b.node("Reshape", [valid_u8, "shape_flat_candidates"], "valid_flat")
    choice = b.node("ArgMax", [valid_flat], "candidate_choice", axis=1, keepdims=0)
    source_choice = b.node("Div", [choice, "eight_i64"], "source_choice")
    transform_choice = b.node("Mod", [choice, "eight_i64"], "transform_choice")
    selected_matrices = b.node(
        "Gather",
        ["dihedral_matrices", transform_choice],
        "selected_matrices",
        axis=0,
    )

    point_groups = _nearest_indices(b, source_coords, source_a, "point_groups")
    point_anchors = b.node("Gather", [source_a, point_groups], "point_anchors", axis=0)
    relative_points = b.node("Sub", [source_coords, point_anchors], "relative_points")
    transformed_points = b.node(
        "Einsum",
        [relative_points, selected_matrices],
        "transformed_points",
        equation="mi,tij->tmj",
    )
    target_for_points = b.node("Unsqueeze", [target_a, "axes_one"], "target_for_points")
    output_coords = b.node("Add", [transformed_points, target_for_points], "output_coords")
    point_groups_row = b.node("Unsqueeze", [point_groups, "axes_zero"], "point_groups_row")
    source_choice_col = b.node("Unsqueeze", [source_choice, "axes_one"], "source_choice_col")
    selected_group = b.node("Equal", [point_groups_row, source_choice_col], "selected_group")
    source_points_row = b.node("Unsqueeze", [is_source, "axes_zero"], "source_points_row")
    selected_source = b.node("And", [selected_group, source_points_row], "selected_source")
    target_valid_col = b.node("Unsqueeze", [target_a_valid, "axes_one"], "target_valid_col")
    selected_points = b.node("And", [selected_source, target_valid_col], "selected_points")

    active_rows = b.node("ReduceMax", ["input"], "active_rows", axes=[1, 3], keepdims=0)
    height = b.node("ArgMin", [active_rows], "height", axis=1, keepdims=0)
    active_cols = b.node("ReduceMax", ["input"], "active_cols", axes=[1, 2], keepdims=0)
    width = b.node("ArgMin", [active_cols], "width", axis=1, keepdims=0)
    selected_points_xy = b.node("Unsqueeze", [selected_points, "axes_two"], "selected_points_xy")
    safe_coords_f16 = b.node(
        "Where",
        [selected_points_xy, output_coords, "dummy_coord_f16"],
        "safe_coords",
    )
    source_colors_row = b.node("Unsqueeze", [source_colors, "axes_zero"], "source_colors_row")
    safe_colors = b.node("Where", [selected_points, source_colors_row, "zero_u8"], "safe_colors")
    flat_coords_f16 = b.node("Reshape", [safe_coords_f16, "shape_coords"], "flat_coords_f16")
    flat_coords = b.node("Cast", [flat_coords_f16], "flat_coords", to=TensorProto.INT64)
    flat_colors = b.node("Reshape", [safe_colors, "shape_values"], "flat_colors")

    valid_rows = b.node("Less", ["row_ids_24", height], "valid_rows")
    valid_cols = b.node("Less", ["col_ids_24", width], "valid_cols")
    valid_domain = b.node("And", [valid_rows, valid_cols], "valid_domain")
    canvas = b.node("Where", [valid_domain, "zero_u8", "sentinel_u8"], "canvas")
    stamped = b.node(
        "ScatterND",
        [canvas, flat_coords, flat_colors],
        "stamped",
        reduction="add",
    )
    padded = b.node("Pad", [stamped, "pads_24_to_30", "sentinel_u8"], "padded", mode="constant")
    b.nodes.append(helper.make_node("Equal", ["channels_u8", padded], ["output"], name="output"))

    graph = helper.make_graph(
        b.nodes,
        "task018_sparse_dihedral",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])],
        b.initializers,
    )
    used_initializers = {name for node in graph.node for name in node.input if name}
    kept_initializers = [init for init in graph.initializer if init.name in used_initializers]
    del graph.initializer[:]
    graph.initializer.extend(kept_initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build_model(), args.out)
    print(args.out)


if __name__ == "__main__":
    main()
