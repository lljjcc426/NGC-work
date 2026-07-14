from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


VARIANTS = {
    "general_k4": (4, "general", False, False),
    "scatter_k4": (4, "scatter", False, False),
    "no_patch_k4_rejected": (4, "none", False, False),
    "scatter_k3_rejected": (3, "scatter", False, False),
    "general_k4_valid29": (4, "general", True, False),
    "scatter_k4_valid29": (4, "scatter", True, False),
    "no_patch_k4_valid29_rejected": (4, "none", True, False),
    "general_k4_valid29_u8period": (4, "general", True, True),
    "scatter_k4_valid29_u8period": (4, "scatter", True, True),
    "scatter_k4_valid29_u8period_probe6": (4, "scatter", True, True),
    "scatter_k4_valid29_u8period_probe6_gather1d_compact": (4, "scatter", True, True),
    "scatter_k4_valid29_u8period_probe6_gather1d_compact_opset20": (4, "scatter", True, True),
    "scatter_k4_valid29_u8period_probe6_gather1d_compact_opset20_u8rows": (4, "scatter", True, True),
    "scatter_k4_valid29_u8period_probe6_gather1d_compact_opset20_u8rows_short78": (4, "scatter", True, True),
    "scatter_k4_valid29_u8period_probe6_gather1d_compact_opset20_u8rows_short78_adaptivek3": (4, "scatter", True, True),
    "scatter_k4_valid29_u8period_probe6_gather1d_compact_opset20_u8rows_short78_adaptivek3_scalaroffsets": (4, "scatter", True, True),
    "scatter_k4_valid29_u8period_probe6_gather1d_compact_opset20_u8rows_short78_adaptivek3_scalaroffsets_addoffsets": (4, "scatter", True, True),
    "scatter_k4_valid29_u8period_probe6_gather1d_compact_opset20_u8rows_short78_adaptivek3_scalaroffsets_addoffsets_thresholdslots": (4, "scatter", True, True),
    "no_patch_k4_valid29_u8period_rejected": (4, "none", True, True),
}


def initializer(name: str, values: object, dtype: np.dtype) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(values, dtype=dtype), name=name)


def build(variant: str) -> onnx.ModelProto:
    copies, fallback, valid29, u8_period = VARIANTS[variant]
    probe_rows = 6 if "_probe6" in variant else 8
    gather_1d = "_gather1d" in variant
    compact = "_compact" in variant
    opset20 = "_opset20" in variant
    u8_rows = "_u8rows" in variant
    short_78 = "_short78" in variant
    adaptive_k3 = "_adaptivek3" in variant
    scalar_offsets = "_scalaroffsets" in variant
    add_offsets = "_addoffsets" in variant
    threshold_slots = variant.endswith("_thresholdslots")
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Conv", ["input", "w_label"], ["labels_f"], name="labels_f"),
        helper.make_node(
            "Cast", ["labels_f"], ["labels"], name="labels", to=TensorProto.UINT8
        ),
    ]
    if gather_1d:
        nodes.append(
            helper.make_node(
                "Gather",
                ["labels", "detect_column"],
                ["detect_strip"],
                name="detect_strip",
                axis=3,
            )
        )
    else:
        nodes.append(
            helper.make_node(
                "Slice",
                ["labels", "detect_starts", "detect_ends", "axes_hw"],
                ["detect_strip"],
                name="detect_strip",
            )
        )
    nodes.extend(
        [
        helper.make_node(
            "Slice",
            ["detect_strip", "probe_starts", "probe_ends", "axis_h"],
            ["detect_probe"],
            name="detect_probe",
        ),
        (
            helper.make_node(
                "Cast",
                ["detect_probe"],
                ["probe_nonzero"],
                name="probe_nonzero",
                to=TensorProto.BOOL,
            )
            if opset20
            else helper.make_node(
                "Greater",
                ["detect_probe", "zero"],
                ["probe_nonzero"],
                name="probe_nonzero",
            )
        ),
        helper.make_node(
            "Where",
            ["probe_nonzero", "detect_probe", "ten"],
            ["probe_cmp"],
            name="probe_cmp",
        ),
        ]
    )
    if short_78:
        nodes.append(
            helper.make_node(
                "Slice",
                ["probe_cmp", "probe_starts", "probe5_ends", "axis_h"],
                ["probe_cmp_5"],
                name="probe_cmp_5",
            )
        )

    # Each aligned pool includes its probe cell, so a larger nonzero value is a
    # compact witness that the candidate period mixes two colors.
    detect_specs = (
        {
            4: (8, 5),
            5: (6, 2),
            6: (5, 1),
            7: (5, 4) if short_78 else (4, 0),
            8: (4, 0) if short_78 else (4, 1),
        }
        if probe_rows == 6
        else {
            4: (8, 7),
            5: (6, 4),
            6: (5, 3),
            7: (4, 0),
            8: (4, 3),
        }
    )
    for period, (kernel, pad_end) in detect_specs.items():
        pool_output = f"detect_pool_{period}"
        nodes.extend(
            [
                helper.make_node(
                    "MaxPool",
                    ["detect_strip"],
                    [pool_output],
                    name=f"detect_pool_{period}",
                    kernel_shape=[kernel] if gather_1d else [kernel, 1],
                    dilations=[period] if gather_1d else [period, 1],
                    pads=[0, pad_end] if gather_1d else [0, 0, pad_end, 0],
                    strides=[1] if gather_1d else [1, 1],
                ),
            ]
        )
        if probe_rows == 6 and period == 7 and not short_78:
            pool_output = "detect_pool_7_aligned"
            nodes.append(
                helper.make_node(
                    "Slice",
                    ["detect_pool_7", "probe_starts", "probe_ends", "axis_h"],
                    [pool_output],
                    name=pool_output,
                )
            )
        nodes.append(
            helper.make_node(
                "Greater",
                [pool_output, "probe_cmp_5" if short_78 and period >= 7 else "probe_cmp"],
                [f"detect_bad_{period}"],
                name=f"detect_bad_{period}",
            )
        )
        if opset20:
            nodes.append(
                helper.make_node(
                    "ReduceMax",
                    [f"detect_bad_{period}"],
                    [f"period_bad_{period}"],
                    name=f"period_bad_{period}",
                    keepdims=0,
                )
            )
        else:
            nodes.extend(
                [
                helper.make_node(
                    "Cast",
                    [f"detect_bad_{period}"],
                    [f"detect_bad_u8_{period}"],
                    name=f"detect_bad_u8_{period}",
                    to=TensorProto.UINT8,
                ),
                helper.make_node(
                    "ReduceMax",
                    [f"detect_bad_u8_{period}"],
                    [f"detect_max_{period}"],
                    name=f"detect_max_{period}",
                    keepdims=0,
                ),
                helper.make_node(
                    "Equal",
                    [f"detect_max_{period}", "zero"],
                    [f"period_ok_{period}"],
                    name=f"period_ok_{period}",
                ),
                ]
            )

    selected = "period_9"
    for period in range(8, 3, -1):
        if period == 4:
            output = "period_u8" if u8_period else "period"
        else:
            output = f"period_from_{period}"
        where_inputs = (
            [f"period_bad_{period}", selected, f"period_{period}"]
            if opset20
            else [f"period_ok_{period}", f"period_{period}", selected]
        )
        nodes.append(helper.make_node("Where", where_inputs, [output], name=output))
        selected = output

    if u8_period and not u8_rows:
        nodes.append(
            helper.make_node(
                "Cast", ["period_u8"], ["period"], name="period", to=TensorProto.INT32
            )
        )

    row_period = "period_u8" if u8_rows else "period"
    if scalar_offsets:
        nodes.extend(
            [
                helper.make_node(
                    "Less", ["period_u8", "period_9"], ["period_lt_9"]
                ),
                helper.make_node(
                    "Less", ["period_u8", "period_6"], ["period_lt_6"]
                ),
            ]
        )
        if threshold_slots:
            nodes.append(
                helper.make_node(
                    "Less", ["period_u8", "period_7"], ["period_lt_7"]
                )
            )
        else:
            nodes.extend(
                [
                    helper.make_node(
                        "Greater", ["period_u8", "period_6"], ["period_gt_6"]
                    ),
                    helper.make_node(
                        "And", ["period_lt_9", "period_gt_6"], ["period_7_to_8"]
                    ),
                ]
            )
        if add_offsets:
            nodes.extend(
                [
                    helper.make_node(
                        "Add", ["period_u8", "period_u8"], ["period_x2"]
                    ),
                    helper.make_node(
                        "Add", ["period_x2", "period_u8"], ["period_x3"]
                    ),
                    helper.make_node(
                        "Where",
                        [
                            "period_lt_9",
                            "offset_0",
                            "period_x3" if threshold_slots else "period_x2",
                        ],
                        ["source_offset_0"],
                    ),
                    helper.make_node(
                        "Where",
                        ["period_lt_6", "period_x2", "period_u8"],
                        ["source_offset_1"],
                    ),
                    helper.make_node(
                        "Where",
                        [
                            "period_lt_7" if threshold_slots else "period_7_to_8",
                            "period_x3" if threshold_slots else "period_x2",
                            "period_x2" if threshold_slots else "period_x3",
                        ],
                        ["source_offset_2"],
                    ),
                ]
            )
        else:
            nodes.extend(
                [
                helper.make_node(
                    "Where",
                    ["period_lt_9", "offset_0", "offset_2"],
                    ["source_multiplier_0"],
                ),
                helper.make_node(
                    "Where",
                    ["period_lt_6", "offset_2", "offset_1"],
                    ["source_multiplier_1"],
                ),
                helper.make_node(
                    "Where",
                    ["period_7_to_8", "offset_2", "offset_3"],
                    ["source_multiplier_2"],
                ),
                *[
                    helper.make_node(
                        "Mul",
                        ["period_u8", f"source_multiplier_{index}"],
                        [f"source_offset_{index}"],
                    )
                    for index in range(3)
                ],
                ]
            )
        nodes.extend(
            [
                helper.make_node(
                    "Add",
                    ["row_index_0", "source_offset_0"],
                    ["source_row_0_raw" if threshold_slots else "source_row_0"],
                ),
                helper.make_node(
                    "Add", ["row_index_0", "source_offset_1"], ["source_row_1"]
                ),
                helper.make_node(
                    "Add",
                    ["row_index_0", "source_offset_2"],
                    ["source_row_2" if threshold_slots else "source_row_2_raw"],
                ),
                helper.make_node(
                    "Less",
                    [
                        "source_row_0_raw" if threshold_slots else "source_row_2_raw",
                        "row_29",
                    ],
                    ["source_row_valid"],
                ),
                helper.make_node(
                    "Where",
                    [
                        "source_row_valid",
                        "source_row_0_raw" if threshold_slots else "source_row_2_raw",
                        "row_index_0",
                    ],
                    ["source_row_0" if threshold_slots else "source_row_2"],
                ),
            ]
        )
        row_index_names = ["source_row_0", "source_row_1", "source_row_2"]
    else:
        nodes.extend(
            [
                helper.make_node("Add", ["row_index_0", row_period], ["row_index_1"]),
                helper.make_node("Add", ["row_index_1", row_period], ["row_index_2"]),
            ]
        )
        row_index_names = ["row_index_0", "row_index_1", "row_index_2"]
        if copies == 4:
            nodes.append(
                helper.make_node(
                    "Add", ["row_index_2", row_period], ["row_index_3_raw"]
                )
            )
            if valid29:
                nodes.extend(
                    [
                        helper.make_node(
                            "Less", ["row_index_3_raw", "row_29"], ["row_index_3_valid"]
                        ),
                        helper.make_node(
                            "Where",
                            ["row_index_3_valid", "row_index_3_raw", "row_index_0"],
                            ["row_index_3"],
                        ),
                    ]
                )
            else:
                nodes.append(
                    helper.make_node(
                        "Min", ["row_index_3_raw", "row_29"], ["row_index_3"]
                    )
                )
            row_index_names.append("row_index_3")
        if adaptive_k3:
            nodes.extend(
                [
                    helper.make_node(
                        "Less", ["period_u8", "period_9"], ["period_lt_9"]
                    ),
                    helper.make_node(
                        "Less", ["period_u8", "period_6"], ["period_lt_6"]
                    ),
                    helper.make_node(
                        "Greater", ["period_u8", "period_6"], ["period_gt_6"]
                    ),
                    helper.make_node(
                        "And", ["period_lt_9", "period_gt_6"], ["period_7_to_8"]
                    ),
                    helper.make_node(
                        "Where",
                        ["period_lt_9", "row_index_0", "row_index_2"],
                        ["source_row_0"],
                    ),
                    helper.make_node(
                        "Where",
                        ["period_lt_6", "row_index_2", "row_index_1"],
                        ["source_row_1"],
                    ),
                    helper.make_node(
                        "Where",
                        ["period_7_to_8", "row_index_2", "row_index_3"],
                        ["source_row_2"],
                    ),
                ]
            )
            row_index_names = ["source_row_0", "source_row_1", "source_row_2"]
    source_rows_output = "source_rows_u8" if u8_rows else "source_rows"
    nodes.append(
        helper.make_node(
            "Concat", row_index_names, [source_rows_output], axis=0, name=source_rows_output
        )
    )
    if u8_rows:
        nodes.append(
            helper.make_node(
                "Cast",
                ["source_rows_u8"],
                ["source_rows"],
                name="source_rows",
                to=TensorProto.INT32,
            )
        )
    nodes.extend(
        [
            helper.make_node(
                "Gather",
                ["labels", "source_rows"],
                ["source_copies"],
                axis=2,
                name="source_copies",
            ),
            helper.make_node(
                "ReduceMax",
                ["source_copies", "reduce_axis_copies"]
                if opset20
                else ["source_copies"],
                ["vertical_classes"],
                **({} if opset20 else {"axes": [2]}),
                keepdims=0,
                name="vertical_classes",
            ),
        ]
    )

    if fallback == "scatter":
        nodes.extend(
            [
                helper.make_node(
                    "GatherND",
                    ["vertical_classes", "patch_indices"],
                    ["patch_current"],
                    name="patch_current",
                ),
                helper.make_node(
                    "GatherND",
                    ["vertical_classes", "patch_fallback_indices"],
                    ["patch_fallback"],
                    name="patch_fallback",
                    **({"batch_dims": 2} if compact else {}),
                ),
                (
                    helper.make_node(
                        "Cast",
                        ["patch_current"],
                        ["patch_has_value"],
                        name="patch_has_value",
                        to=TensorProto.BOOL,
                    )
                    if opset20
                    else helper.make_node(
                        "Greater",
                        ["patch_current", "zero"],
                        ["patch_has_value"],
                        name="patch_has_value",
                    )
                ),
                helper.make_node(
                    "Where",
                    ["patch_has_value", "patch_current", "patch_fallback"],
                    ["patch_updates"],
                    name="patch_updates",
                ),
                helper.make_node(
                    "ScatterND",
                    ["vertical_classes", "patch_indices", "patch_updates"],
                    ["corrected_classes"],
                    name="corrected_classes",
                ),
                helper.make_node(
                    "Pad",
                    [
                        "corrected_classes",
                        "pad_bottom_right" if valid29 else "pad_bottom",
                    ],
                    ["class_table"],
                    name="class_table",
                    mode="constant",
                ),
            ]
        )
    elif fallback == "general":
        primary_classes = "vertical_classes"
        if not valid29:
            nodes.append(
                helper.make_node(
                    "Slice",
                    ["vertical_classes", "crop_starts", "crop_ends", "axis_w"],
                    ["primary_classes"],
                    name="primary_classes",
                )
            )
            primary_classes = "primary_classes"
        nodes.extend(
            [
                helper.make_node(
                    "Gather",
                    ["vertical_classes", "fallback_columns"],
                    ["fallback_classes"],
                    axis=3,
                    name="fallback_classes",
                ),
                helper.make_node(
                    "Greater",
                    [primary_classes, "zero"],
                    ["primary_nonzero"],
                    name="primary_nonzero",
                ),
                helper.make_node(
                    "Where",
                    ["primary_nonzero", primary_classes, "fallback_classes"],
                    ["corrected_classes_29"],
                    name="corrected_classes_29",
                ),
                helper.make_node(
                    "Pad",
                    ["corrected_classes_29", "pad_bottom_right"],
                    ["class_table"],
                    name="class_table",
                    mode="constant",
                ),
            ]
        )
    else:
        nodes.append(
            helper.make_node(
                "Pad",
                [
                    "vertical_classes",
                    "pad_bottom_right" if valid29 else "pad_bottom",
                ],
                ["class_table"],
                name="class_table",
                mode="constant",
            )
        )

    output_rows = "output_rows_u8" if u8_period else "output_rows"
    output_rows_30 = "output_rows_30_u8" if u8_period else "output_rows_30"
    output_period = "period_u8" if u8_period else "period"
    border_source = "period_9" if compact else "border_row"
    nodes.extend(
        [
            helper.make_node("Mod", ["rows_0_28", output_period], [output_rows]),
            helper.make_node(
                "Concat", [output_rows, border_source], [output_rows_30], axis=0
            ),
        ]
    )
    if u8_period:
        nodes.append(
            helper.make_node(
                "Cast",
                [output_rows_30],
                ["output_rows_30"],
                name="output_rows_30",
                to=TensorProto.INT32,
            )
        )
    nodes.extend(
        [
            helper.make_node(
                "Gather",
                ["class_table", "output_rows_30"],
                ["label_grid"],
                axis=2,
            ),
            helper.make_node("Equal", ["label_grid", "colors"], ["output"]),
        ]
    )

    label_weights = np.zeros((1, 10, 2, 2), dtype=np.float32)
    label_weights[0, :, 0, 0] = np.arange(10, dtype=np.float32)
    if not valid29:
        label_weights = label_weights[:, :, :1, :1]

    period_dtype = np.uint8 if u8_period else np.int32
    output_index_dtype = np.uint8 if u8_period else np.int32
    initializers = [
        initializer(
            "w_label",
            label_weights,
            np.float32,
        ),
        initializer(
            "colors",
            np.asarray([10, *range(1, 10)], dtype=np.uint8).reshape(1, 10, 1, 1),
            np.uint8,
        ),
        initializer("zero", 0, np.uint8),
        initializer("ten", 10, np.uint8),
        initializer("detect_starts", [0, 21], np.int64),
        initializer("detect_ends", [29, 22], np.int64),
        initializer("axes_hw", [2, 3], np.int64),
        initializer("detect_column", 21, np.int64),
        initializer("probe_starts", [0], np.int64),
        initializer("probe_ends", [probe_rows], np.int64),
        initializer("probe5_ends", [5], np.int64),
        initializer("axis_h", [2], np.int64),
        *[
            initializer(
                f"period_{period}",
                [period] if compact and period == 9 else period,
                period_dtype,
            )
            for period in range(4, 10)
        ],
        initializer(
            "row_index_0",
            np.arange(9).reshape(1, 9),
            np.uint8 if u8_rows else np.int32,
        ),
        initializer("row_29", 29, np.uint8 if u8_rows else np.int32),
        *[
            initializer(f"offset_{offset}", offset, np.uint8)
            for offset in range(4)
        ],
        initializer("reduce_axis_copies", [2], np.int64),
        initializer("pad_bottom", [0, 0, 0, 0, 0, 0, 1, 0], np.int64),
        initializer("pad_bottom_right", [0, 0, 0, 0, 0, 0, 1, 1], np.int64),
        initializer("rows_0_28", np.arange(29), output_index_dtype),
        initializer("border_row", [9], output_index_dtype),
    ]
    if fallback == "scatter":
        patch_indices = [[0, 0, 2, 3], [0, 0, 2, 4]]
        patch_fallback_indices = [[0, 0, 2, 12], [0, 0, 2, 13]]
        if compact:
            patch_indices = [[patch_indices]]
            patch_fallback_indices = [[[[2, 12], [2, 13]]]]
        initializers.extend(
            [
                initializer("patch_indices", patch_indices, np.int64),
                initializer("patch_fallback_indices", patch_fallback_indices, np.int64),
            ]
        )
    elif fallback == "general":
        if not valid29:
            initializers.extend(
                [
                initializer("crop_starts", [0], np.int64),
                initializer("crop_ends", [29], np.int64),
                initializer("axis_w", [3], np.int64),
                ]
            )
        initializers.extend(
            [
                initializer(
                    "fallback_columns",
                    [min(column + 9, 28 if valid29 else 29) for column in range(29)],
                    np.int32,
                ),
            ]
        )

    used_initializers = {
        name for node in nodes for name in node.input if name
    }
    initializers = [item for item in initializers if item.name in used_initializers]

    graph = helper.make_graph(
        nodes,
        f"task110_{variant}",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 20 if opset20 else 13)],
        producer_name=f"ngc_e_task110_20260713_{variant}",
    )
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variant", choices=[*VARIANTS, "all"], default="all")
    args = parser.parse_args()

    variants = VARIANTS if args.variant == "all" else [args.variant]
    for variant in variants:
        output = args.output_root / variant / "task110.onnx"
        output.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(build(variant), output)
        print(output)


if __name__ == "__main__":
    main()
