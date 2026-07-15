#!/usr/bin/env python
"""High-yield task233 rewrites derived from the public code-golf solver."""
from __future__ import annotations

import copy
import csv
import math
import pathlib
import shutil
import sys
import tempfile
from dataclasses import dataclass

import numpy as np
import onnx
import onnxruntime
from onnx import TensorProto, helper, numpy_helper


NGC_ROOT = pathlib.Path(r"F:\kaggle\neurogolf-2026")
WORKPLACE = pathlib.Path(__file__).resolve().parent
BASE_MODEL = (
    NGC_ROOT
    / "work"
    / "e_high737801_recovered7_v1"
    / "models"
    / "task233.onnx"
)
OUT_ROOT = WORKPLACE / "optimized_onnx"
OUT_CSV = WORKPLACE / "e_task233_20260713_candidates.csv"
BEST_DIR = OUT_ROOT / "task233_20260713_best"

sys.path.insert(0, str(NGC_ROOT / "data" / "neurogolf_utils"))
import neurogolf_utils as ng  # noqa: E402


ng._NEUROGOLF_DIR = str((NGC_ROOT / "data").resolve()) + "\\"
onnxruntime.set_default_logger_severity(3)


@dataclass
class Verification:
    ok: bool
    arc_agi_pass: int
    arc_agi_fail: int
    arc_gen_pass: int
    arc_gen_fail: int
    first_error: str = ""


def set_initializer(model: onnx.ModelProto, name: str, value: np.ndarray) -> None:
    kept = [item for item in model.graph.initializer if item.name != name]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.initializer.append(numpy_helper.from_array(value, name=name))


def replace_nodes_by_range(
    model: onnx.ModelProto,
    first_output: str,
    last_output: str,
    replacements: list[onnx.NodeProto],
) -> None:
    nodes = list(model.graph.node)
    start = next(
        index
        for index, node in enumerate(nodes)
        if first_output in node.output
    )
    end = next(
        index
        for index, node in enumerate(nodes[start:], start)
        if last_output in node.output
    )
    nodes[start : end + 1] = replacements
    del model.graph.node[:]
    model.graph.node.extend(nodes)


def prune(model: onnx.ModelProto) -> None:
    needed = {item.name for item in model.graph.output}
    kept_reversed: list[onnx.NodeProto] = []
    for node in reversed(model.graph.node):
        if any(name in needed for name in node.output):
            kept_reversed.append(node)
            needed.update(name for name in node.input if name)
    kept_outputs = {tuple(node.output) for node in kept_reversed}
    kept = [node for node in model.graph.node if tuple(node.output) in kept_outputs]
    used = {name for node in kept for name in node.input if name}
    initializers = [item for item in model.graph.initializer if item.name in used]
    del model.graph.node[:]
    model.graph.node.extend(kept)
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)


def remove_value_info(model: onnx.ModelProto, names: set[str]) -> None:
    kept = [item for item in model.graph.value_info if item.name not in names]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept)


def set_value_info(
    model: onnx.ModelProto, name: str, dtype: int, shape: list[int]
) -> None:
    remove_value_info(model, {name})
    model.graph.value_info.append(helper.make_tensor_value_info(name, dtype, shape))


def finalize(model: onnx.ModelProto) -> onnx.ModelProto:
    prune(model)
    live = {name for node in model.graph.node for name in node.output if name}
    kept = [item for item in model.graph.value_info if item.name in live]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    return model


def cast_target(node: onnx.NodeProto, dtype: int) -> None:
    del node.attribute[:]
    node.attribute.append(helper.make_attribute("to", dtype))


def rewrite_topk_uint8(model: onnx.ModelProto) -> None:
    targets = {"safe_name_53", "safe_name_106"}
    found: set[str] = set()
    for node in model.graph.node:
        if node.op_type == "Cast" and node.output and node.output[0] in targets:
            cast_target(node, TensorProto.UINT8)
            found.add(node.output[0])
    if found != targets:
        raise RuntimeError(f"missing TopK casts: {sorted(targets - found)}")
    topk_values = {
        node.output[0]
        for node in model.graph.node
        if node.op_type == "TopK" and node.input[0] in targets
    }
    remove_value_info(model, targets | topk_values)


def rewrite_axis_binary_reduction(model: onnx.ModelProto) -> None:
    replacements = [
        helper.make_node(
            "ReduceMax",
            ["safe_name_74"],
            ["task233_row_values"],
            name="task233_row_values",
            axes=[3],
            keepdims=0,
        ),
        helper.make_node(
            "Min",
            ["task233_row_values", "safe_name_42"],
            ["safe_name_76"],
            name="safe_name_76",
        ),
        helper.make_node(
            "Cast",
            ["safe_name_76"],
            ["safe_name_77"],
            name="safe_name_77",
            to=TensorProto.FLOAT16,
        ),
        helper.make_node(
            "ReduceMax",
            ["safe_name_74"],
            ["task233_col_values"],
            name="task233_col_values",
            axes=[2],
            keepdims=0,
        ),
        helper.make_node(
            "Min",
            ["task233_col_values", "safe_name_42"],
            ["safe_name_78"],
            name="safe_name_78",
        ),
        helper.make_node(
            "Cast",
            ["safe_name_78"],
            ["safe_name_79"],
            name="safe_name_79",
            to=TensorProto.FLOAT16,
        ),
    ]
    replace_nodes_by_range(model, "safe_name_75", "safe_name_79", replacements)


def rewrite_axis_extents(model: onnx.ModelProto) -> None:
    set_initializer(model, "task233_one_i32", np.asarray([1], dtype=np.int32))
    replacements = [
        helper.make_node(
            "ArgMax",
            ["safe_name_76"],
            ["safe_name_80"],
            name="safe_name_80",
            axis=2,
            keepdims=0,
            select_last_index=0,
        ),
        helper.make_node(
            "Cast",
            ["safe_name_80"],
            ["safe_name_81"],
            name="safe_name_81",
            to=TensorProto.INT32,
        ),
        helper.make_node(
            "ArgMax",
            ["safe_name_76"],
            ["task233_last_row_i64"],
            name="task233_last_row_i64",
            axis=2,
            keepdims=0,
            select_last_index=1,
        ),
        helper.make_node(
            "Cast",
            ["task233_last_row_i64"],
            ["task233_last_row_i32"],
            name="task233_last_row_i32",
            to=TensorProto.INT32,
        ),
        helper.make_node(
            "Sub",
            ["task233_last_row_i32", "safe_name_81"],
            ["task233_row_span"],
            name="task233_row_span",
        ),
        helper.make_node(
            "Add",
            ["task233_row_span", "task233_one_i32"],
            ["task233_row_extent_i32"],
            name="task233_row_extent_i32",
        ),
        helper.make_node(
            "Cast",
            ["task233_row_extent_i32"],
            ["safe_name_84"],
            name="safe_name_84",
            to=TensorProto.FLOAT16,
        ),
        helper.make_node(
            "ReduceMax",
            ["safe_name_74"],
            ["task233_col_values"],
            name="task233_col_values",
            axes=[2],
            keepdims=0,
        ),
        helper.make_node(
            "Min",
            ["task233_col_values", "safe_name_42"],
            ["safe_name_78"],
            name="safe_name_78",
        ),
        helper.make_node(
            "ArgMax",
            ["safe_name_78"],
            ["safe_name_82"],
            name="safe_name_82",
            axis=2,
            keepdims=0,
            select_last_index=0,
        ),
        helper.make_node(
            "Cast",
            ["safe_name_82"],
            ["safe_name_83"],
            name="safe_name_83",
            to=TensorProto.INT32,
        ),
        helper.make_node(
            "ArgMax",
            ["safe_name_78"],
            ["task233_last_col_i64"],
            name="task233_last_col_i64",
            axis=2,
            keepdims=0,
            select_last_index=1,
        ),
        helper.make_node(
            "Cast",
            ["task233_last_col_i64"],
            ["task233_last_col_i32"],
            name="task233_last_col_i32",
            to=TensorProto.INT32,
        ),
        helper.make_node(
            "Sub",
            ["task233_last_col_i32", "safe_name_83"],
            ["task233_col_span"],
            name="task233_col_span",
        ),
        helper.make_node(
            "Add",
            ["task233_col_span", "task233_one_i32"],
            ["task233_col_extent_i32"],
            name="task233_col_extent_i32",
        ),
        helper.make_node(
            "Cast",
            ["task233_col_extent_i32"],
            ["safe_name_85"],
            name="safe_name_85",
            to=TensorProto.FLOAT16,
        ),
    ]
    replace_nodes_by_range(model, "safe_name_77", "safe_name_85", replacements)


def rewrite_compact_updates(model: onnx.ModelProto) -> None:
    replacements = [
        helper.make_node(
            "Where",
            ["task233_active_templates", "safe_name_68", "safe_name_31"],
            ["task233_active_colors"],
            name="task233_active_colors",
        ),
        helper.make_node(
            "Unsqueeze",
            ["task233_active_colors", "task233_axis0"],
            ["task233_template_colors"],
            name="task233_template_colors",
        ),
        helper.make_node(
            "Where",
            ["task233_generic_holes", "safe_name_31", "task233_template_colors"],
            ["task233_generic_updates_masked"],
            name="task233_generic_updates_masked",
        ),
    ]
    replace_nodes_by_range(
        model,
        "task233_template_colors",
        "task233_generic_updates_masked",
        replacements,
    )
    index_concat = next(node for node in model.graph.node if "safe_name_266" in node.output)
    update_concat = next(node for node in model.graph.node if "safe_name_267" in node.output)
    index_concat.input.remove("safe_name_30")
    update_concat.input.remove("safe_name_31")
    remove_value_info(model, {"safe_name_266", "safe_name_267"})


def drop_special_branches(model: onnx.ModelProto, branches: set[str]) -> None:
    branch_tensors = {
        "36": ("safe_name_252", "safe_name_34"),
        "9": ("safe_name_258", "safe_name_259"),
        "27": ("safe_name_264", "safe_name_265"),
    }
    index_concat = next(node for node in model.graph.node if "safe_name_266" in node.output)
    update_concat = next(node for node in model.graph.node if "safe_name_267" in node.output)
    for branch in branches:
        index_name, update_name = branch_tensors[branch]
        index_concat.input.remove(index_name)
        update_concat.input.remove(update_name)
    remove_value_info(model, {"safe_name_266", "safe_name_267"})


def rewrite_key_nonzero(model: onnx.ModelProto) -> None:
    """List the 1-5 learned template blocks in row-major order without TopK."""
    set_initializer(model, "task233_key_sentinels", np.ones(5, dtype=np.bool_))
    set_initializer(model, "task233_key_limit", np.asarray([784], dtype=np.int64))
    set_initializer(model, "task233_zero_i64", np.asarray([0], dtype=np.int64))
    set_initializer(model, "task233_axis0", np.asarray([0], dtype=np.int64))
    replacements = [
        helper.make_node(
            "Concat",
            ["safe_name_52", "task233_key_sentinels"],
            ["task233_key_padded"],
            name="task233_key_padded",
            axis=0,
        ),
        helper.make_node(
            "NonZero",
            ["task233_key_padded"],
            ["task233_key_nonzero"],
            name="task233_key_nonzero",
        ),
        helper.make_node(
            "Squeeze",
            ["task233_key_nonzero", "task233_axis0"],
            ["task233_key_nonzero_flat"],
            name="task233_key_nonzero_flat",
        ),
        helper.make_node(
            "Slice",
            ["task233_key_nonzero_flat", "task233_axis0", "safe_name_3", "task233_axis0"],
            ["task233_key_indices_raw"],
            name="task233_key_indices_raw",
        ),
        helper.make_node(
            "Less",
            ["task233_key_indices_raw", "task233_key_limit"],
            ["safe_name_57"],
            name="safe_name_57",
        ),
        helper.make_node(
            "Where",
            ["safe_name_57", "task233_key_indices_raw", "task233_zero_i64"],
            ["safe_name_55"],
            name="safe_name_55",
        ),
        helper.make_node(
            "Cast",
            ["safe_name_55"],
            ["safe_name_56"],
            name="safe_name_56",
            to=TensorProto.INT32,
        ),
    ]
    replace_nodes_by_range(model, "safe_name_53", "safe_name_57", replacements)
    set_value_info(model, "task233_key_nonzero", TensorProto.INT64, [1, 10])


def rewrite_direct_scan_tail(
    model: onnx.ModelProto, *, conflict_aware: bool = False
) -> None:
    """Use the code-golf solver's first row-major exact match per template."""
    set_initializer(model, "task233_axis0", np.asarray([0], dtype=np.int64))
    set_initializer(model, "task233_shape_1x45", np.asarray([1, 45], dtype=np.int64))

    direct_nodes = [
        helper.make_node(
            "Cast",
            ["safe_name_105"],
            ["task233_match_u8"],
            name="task233_match_u8",
            to=TensorProto.UINT8,
        ),
        helper.make_node(
            "ReduceMax",
            ["task233_match_u8"],
            ["task233_match_exists"],
            name="task233_match_exists",
            axes=[1],
            keepdims=1,
        ),
        helper.make_node(
            "Cast",
            ["task233_match_exists"],
            ["task233_match_valid"],
            name="task233_match_valid",
            to=TensorProto.BOOL,
        ),
        helper.make_node(
            "And",
            ["task233_match_valid", "safe_name_58"],
            ["task233_active_templates"],
            name="task233_active_templates",
        ),
        helper.make_node(
            "ArgMax",
            ["task233_match_u8"],
            ["task233_match_index_i64"],
            name="task233_match_index_i64",
            axis=1,
            keepdims=1,
            select_last_index=0,
        ),
        helper.make_node(
            "Cast",
            ["task233_match_index_i64"],
            ["task233_match_index_i32"],
            name="task233_match_index_i32",
            to=TensorProto.INT32,
        ),
        helper.make_node(
            "Div",
            ["task233_match_index_i32", "safe_name_18"],
            ["task233_match_rows"],
            name="task233_match_rows",
        ),
        helper.make_node(
            "Mul",
            ["task233_match_rows", "safe_name_5"],
            ["task233_match_row_gaps"],
            name="task233_match_row_gaps",
        ),
        helper.make_node(
            "Add",
            ["task233_match_index_i32", "task233_match_row_gaps"],
            ["task233_match_bases"],
            name="task233_match_bases",
        ),
        helper.make_node(
            "Add",
            ["task233_match_bases", "safe_name_17"],
            ["task233_generic_indices_raw"],
            name="task233_generic_indices_raw",
        ),
        helper.make_node(
            "Where",
            ["task233_active_templates", "task233_generic_indices_raw", "safe_name_30"],
            ["task233_generic_indices_masked"],
            name="task233_generic_indices_masked",
        ),
        helper.make_node(
            "Reshape",
            ["safe_name_92", "safe_name_23"],
            ["safe_name_115"],
            name="safe_name_115",
        ),
        helper.make_node(
            "Gather",
            ["safe_name_115", "task233_generic_indices_masked"],
            ["task233_generic_patch"],
            name="task233_generic_patch",
            axis=1,
        ),
        helper.make_node(
            "Equal",
            ["task233_generic_patch", "safe_name_29"],
            ["task233_generic_holes"],
            name="task233_generic_holes",
        ),
        helper.make_node(
            "Unsqueeze",
            ["safe_name_68", "task233_axis0"],
            ["task233_template_colors"],
            name="task233_template_colors",
        ),
        helper.make_node(
            "Where",
            ["task233_generic_holes", "safe_name_31", "task233_template_colors"],
            ["task233_generic_updates_raw"],
            name="task233_generic_updates_raw",
        ),
        helper.make_node(
            "Unsqueeze",
            ["task233_active_templates", "task233_axis0"],
            ["task233_active_templates_3d"],
            name="task233_active_templates_3d",
        ),
        helper.make_node(
            "Where",
            ["task233_active_templates_3d", "task233_generic_updates_raw", "safe_name_31"],
            ["task233_generic_updates_masked"],
            name="task233_generic_updates_masked",
        ),
        helper.make_node(
            "Reshape",
            ["task233_generic_indices_masked", "task233_shape_1x45"],
            ["task233_generic_indices"],
            name="task233_generic_indices",
        ),
        helper.make_node(
            "Reshape",
            ["task233_generic_updates_masked", "task233_shape_1x45"],
            ["task233_generic_updates"],
            name="task233_generic_updates",
        ),
    ]
    if conflict_aware:
        for opset in model.opset_import:
            if opset.domain in {"", "ai.onnx"}:
                opset.version = max(opset.version, 14)
        set_initializer(model, "task233_axis1", np.asarray([1], dtype=np.int64))
        set_initializer(model, "task233_overlap_upper", np.asarray([3], dtype=np.int8))
        set_initializer(model, "task233_overlap_lower", np.asarray([-3], dtype=np.int8))
        set_initializer(model, "task233_inactive_coord", np.asarray([127], dtype=np.int8))
        set_initializer(model, "task233_off_diagonal", ~np.eye(5, dtype=np.bool_))
        selection_nodes = [
            helper.make_node(
                "ArgMax",
                ["task233_match_u8"],
                ["task233_match_first_i64"],
                name="task233_match_first_i64",
                axis=1,
                keepdims=1,
                select_last_index=0,
            ),
            helper.make_node(
                "Cast",
                ["task233_match_first_i64"],
                ["task233_match_first_i32"],
                name="task233_match_first_i32",
                to=TensorProto.INT32,
            ),
            helper.make_node(
                "ArgMax",
                ["task233_match_u8"],
                ["task233_match_last_i64"],
                name="task233_match_last_i64",
                axis=1,
                keepdims=1,
                select_last_index=1,
            ),
            helper.make_node(
                "Cast",
                ["task233_match_last_i64"],
                ["task233_match_last_i32"],
                name="task233_match_last_i32",
                to=TensorProto.INT32,
            ),
            helper.make_node(
                "Div",
                ["task233_match_first_i32", "safe_name_18"],
                ["task233_match_first_rows_i32"],
                name="task233_match_first_rows_i32",
            ),
            helper.make_node(
                "Cast",
                ["task233_match_first_rows_i32"],
                ["task233_match_first_rows"],
                name="task233_match_first_rows",
                to=TensorProto.INT8,
            ),
            helper.make_node(
                "Mod",
                ["task233_match_first_i32", "safe_name_18"],
                ["task233_match_first_cols_i32"],
                name="task233_match_first_cols_i32",
            ),
            helper.make_node(
                "Cast",
                ["task233_match_first_cols_i32"],
                ["task233_match_first_cols"],
                name="task233_match_first_cols",
                to=TensorProto.INT8,
            ),
            helper.make_node(
                "Where",
                ["task233_active_templates", "task233_match_first_rows", "task233_inactive_coord"],
                ["task233_match_active_rows"],
                name="task233_match_active_rows",
            ),
            helper.make_node(
                "Where",
                ["task233_active_templates", "task233_match_first_cols", "task233_inactive_coord"],
                ["task233_match_active_cols"],
                name="task233_match_active_cols",
            ),
            helper.make_node(
                "Transpose",
                ["task233_match_active_rows"],
                ["task233_match_first_rows_t"],
                name="task233_match_first_rows_t",
                perm=[1, 0],
            ),
            helper.make_node(
                "Transpose",
                ["task233_match_active_cols"],
                ["task233_match_first_cols_t"],
                name="task233_match_first_cols_t",
                perm=[1, 0],
            ),
            helper.make_node(
                "Sub",
                ["task233_match_active_rows", "task233_match_first_rows_t"],
                ["task233_match_row_delta"],
                name="task233_match_row_delta",
            ),
            helper.make_node(
                "Greater",
                ["task233_match_row_delta", "task233_overlap_lower"],
                ["task233_match_row_above_lower"],
                name="task233_match_row_above_lower",
            ),
            helper.make_node(
                "Less",
                ["task233_match_row_delta", "task233_overlap_upper"],
                ["task233_match_row_below_upper"],
                name="task233_match_row_below_upper",
            ),
            helper.make_node(
                "And",
                ["task233_match_row_above_lower", "task233_match_row_below_upper"],
                ["task233_match_row_overlap"],
                name="task233_match_row_overlap",
            ),
            helper.make_node(
                "Sub",
                ["task233_match_active_cols", "task233_match_first_cols_t"],
                ["task233_match_col_delta"],
                name="task233_match_col_delta",
            ),
            helper.make_node(
                "Greater",
                ["task233_match_col_delta", "task233_overlap_lower"],
                ["task233_match_col_above_lower"],
                name="task233_match_col_above_lower",
            ),
            helper.make_node(
                "Less",
                ["task233_match_col_delta", "task233_overlap_upper"],
                ["task233_match_col_below_upper"],
                name="task233_match_col_below_upper",
            ),
            helper.make_node(
                "And",
                ["task233_match_col_above_lower", "task233_match_col_below_upper"],
                ["task233_match_col_overlap"],
                name="task233_match_col_overlap",
            ),
            helper.make_node(
                "And",
                ["task233_match_row_overlap", "task233_match_col_overlap"],
                ["task233_match_spatial_overlap"],
                name="task233_match_spatial_overlap",
            ),
            helper.make_node(
                "And",
                ["task233_match_spatial_overlap", "task233_off_diagonal"],
                ["task233_active_other_overlap"],
                name="task233_active_other_overlap",
            ),
            helper.make_node(
                "Cast",
                ["task233_active_other_overlap"],
                ["task233_active_other_overlap_u8"],
                name="task233_active_other_overlap_u8",
                to=TensorProto.UINT8,
            ),
            helper.make_node(
                "ReduceMax",
                ["task233_active_other_overlap_u8"],
                ["task233_has_other_overlap_u8"],
                name="task233_has_other_overlap_u8",
                axes=[1],
                keepdims=1,
            ),
            helper.make_node(
                "Cast",
                ["task233_has_other_overlap_u8"],
                ["task233_has_other_overlap"],
                name="task233_has_other_overlap",
                to=TensorProto.BOOL,
            ),
            helper.make_node(
                "Where",
                ["task233_has_other_overlap", "task233_match_last_i32", "task233_match_first_i32"],
                ["task233_match_index_i32"],
                name="task233_match_index_i32",
            ),
        ]
        direct_nodes[4:6] = selection_nodes
    replace_nodes_by_range(model, "safe_name_106", "safe_name_247", direct_nodes)

    index_concat = next(node for node in model.graph.node if "safe_name_266" in node.output)
    update_concat = next(node for node in model.graph.node if "safe_name_267" in node.output)
    del index_concat.input[:]
    index_concat.input.extend(
        [
            "task233_generic_indices",
            "safe_name_30",
            "safe_name_252",
            "safe_name_258",
            "safe_name_264",
        ]
    )
    del update_concat.input[:]
    update_concat.input.extend(
        [
            "task233_generic_updates",
            "safe_name_31",
            "safe_name_34",
            "safe_name_259",
            "safe_name_265",
        ]
    )


def drop_valid_top_left_mask(model: onnx.ModelProto) -> None:
    masked = next(
        node for node in model.graph.node if "task233_masked_codes" in node.output
    )
    old = masked.output[0]
    replacement = masked.input[1]
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == old:
                node.input[index] = replacement


def score_cost(model: onnx.ModelProto) -> tuple[int, int, int] | None:
    sanitized = ng.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        print("score_cost failed: sanitize_model returned None", file=sys.stderr)
        return None
    with tempfile.TemporaryDirectory() as tmp:
        options = onnxruntime.SessionOptions()
        options.enable_profiling = True
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.log_severity_level = 3
        options.profile_file_prefix = str(pathlib.Path(tmp) / "task233")
        try:
            session = onnxruntime.InferenceSession(sanitized.SerializeToString(), options)
            example = ng.convert_to_numpy(ng.load_examples(233)["train"][0])
            if example is None:
                return None
            ng.run_network(session, example["input"])
            trace_path = session.end_profiling()
            memory, params = ng.score_network(sanitized, trace_path)
        except Exception as exc:
            print(f"score_cost failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return None
    if memory is None or params is None or memory < 0 or params < 0:
        print(f"score_cost failed: memory={memory} params={params}", file=sys.stderr)
        return None
    return int(memory), int(params), int(memory) + int(params)


def verify_all(model: onnx.ModelProto) -> Verification:
    sanitized = ng.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return Verification(False, 0, 4, 0, 262, "sanitize_model returned None")
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    try:
        session = onnxruntime.InferenceSession(sanitized.SerializeToString(), options)
    except Exception as exc:
        return Verification(False, 0, 4, 0, 262, f"session: {type(exc).__name__}: {exc}")

    counts = {"arc_agi_pass": 0, "arc_agi_fail": 0, "arc_gen_pass": 0, "arc_gen_fail": 0}
    first_error = ""
    examples = ng.load_examples(233)
    for split, subset in (("train", examples["train"]), ("test", examples["test"]), ("arc-gen", examples["arc-gen"])):
        for index, example in enumerate(subset):
            batch = ng.convert_to_numpy(example)
            if batch is None:
                continue
            key = "arc_gen" if split == "arc-gen" else "arc_agi"
            try:
                actual = ng.run_network(session, batch["input"])
                correct = np.array_equal(actual, batch["output"])
            except Exception as exc:
                correct = False
                if not first_error:
                    first_error = f"{split}[{index}] runtime {type(exc).__name__}: {exc}"
            if correct:
                counts[f"{key}_pass"] += 1
            else:
                counts[f"{key}_fail"] += 1
                if not first_error:
                    mismatch = int(np.count_nonzero(actual != batch["output"]))
                    first_error = f"{split}[{index}] mismatch_elements={mismatch}"
    ok = counts["arc_agi_fail"] == 0 and counts["arc_gen_fail"] == 0
    return Verification(ok=ok, first_error=first_error, **counts)


def points(cost: int) -> float:
    return max(1.0, 25.0 - math.log(max(1, cost)))


def build_candidates(base: onnx.ModelProto):
    unsupported_topk = copy.deepcopy(base)
    rewrite_topk_uint8(unsupported_topk)
    yield "topk_uint8_unsupported", finalize(unsupported_topk)

    axis = copy.deepcopy(base)
    rewrite_axis_binary_reduction(axis)
    yield "axis_min", finalize(axis)

    direct = copy.deepcopy(axis)
    rewrite_direct_scan_tail(direct)
    yield "codegolf_direct_scan", finalize(direct)

    no_valid_mask = copy.deepcopy(direct)
    drop_valid_top_left_mask(no_valid_mask)
    yield "codegolf_direct_scan_no_valid_mask", finalize(no_valid_mask)

    conflict = copy.deepcopy(axis)
    rewrite_direct_scan_tail(conflict, conflict_aware=True)
    yield "codegolf_conflict_scan", finalize(conflict)

    conflict_no_valid_mask = copy.deepcopy(conflict)
    drop_valid_top_left_mask(conflict_no_valid_mask)
    yield "codegolf_conflict_scan_no_valid_mask", finalize(conflict_no_valid_mask)

    nonzero_keys = copy.deepcopy(conflict_no_valid_mask)
    rewrite_key_nonzero(nonzero_keys)
    yield "codegolf_conflict_nonzero_keys", finalize(nonzero_keys)

    compact = copy.deepcopy(conflict_no_valid_mask)
    rewrite_axis_extents(compact)
    rewrite_compact_updates(compact)
    yield "codegolf_compact", finalize(compact)

    for branch in ("36", "9", "27"):
        ablated = copy.deepcopy(compact)
        drop_special_branches(ablated, {branch})
        yield f"codegolf_compact_drop_special_{branch}", finalize(ablated)

    no_specials = copy.deepcopy(compact)
    drop_special_branches(no_specials, {"36", "9", "27"})
    yield "codegolf_compact_drop_all_specials", finalize(no_specials)


def main() -> int:
    base = finalize(onnx.load(BASE_MODEL))
    base_score = score_cost(base)
    if base_score is None:
        raise RuntimeError("could not score baseline")
    base_memory, base_params, base_cost = base_score

    rows: list[dict[str, object]] = []
    accepted: list[tuple[int, str, pathlib.Path]] = []
    for name, model in build_candidates(base):
        score = score_cost(model)
        verification = verify_all(model)
        if score is None:
            memory = params = cost = None
            status = "rejected_score_error"
        else:
            memory, params, cost = score
            if not verification.ok:
                status = "rejected_wrong"
            elif cost >= base_cost:
                status = "rejected_not_better"
            else:
                status = "accepted"
                out_dir = OUT_ROOT / f"task233_20260713_candidate_{name}"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / "task233.onnx"
                onnx.save(model, out_path)
                accepted.append((cost, name, out_path))
        rows.append(
            {
                "candidate": name,
                "status": status,
                "base_memory": base_memory,
                "base_params": base_params,
                "base_cost": base_cost,
                "memory": "" if score is None else memory,
                "params": "" if score is None else params,
                "cost": "" if score is None else cost,
                "delta_cost": "" if score is None else base_cost - cost,
                "points": "" if score is None else f"{points(cost):.9f}",
                "delta_points": "" if score is None else f"{points(cost) - points(base_cost):.9f}",
                "arc_agi_pass": verification.arc_agi_pass,
                "arc_agi_fail": verification.arc_agi_fail,
                "arc_gen_pass": verification.arc_gen_pass,
                "arc_gen_fail": verification.arc_gen_fail,
                "nodes": len(model.graph.node),
                "initializers": len(model.graph.initializer),
                "first_error": verification.first_error,
            }
        )
        print(
            f"{name}: {status}, cost={cost}, "
            f"arc_agi={verification.arc_agi_pass}/{verification.arc_agi_pass + verification.arc_agi_fail}, "
            f"arc_gen={verification.arc_gen_pass}/{verification.arc_gen_pass + verification.arc_gen_fail}",
            flush=True,
        )

    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if not accepted:
        print(f"No accepted candidate. Wrote {OUT_CSV}")
        return 1
    best_cost, best_name, best_path = min(accepted)
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best_path, BEST_DIR / "task233.onnx")
    print(f"Best={best_name} cost={best_cost} path={BEST_DIR / 'task233.onnx'}")
    print(f"Wrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
