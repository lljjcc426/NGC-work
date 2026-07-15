#!/usr/bin/env python
"""Build and fully score structural task118 rewrites.

The accepted family encodes the arm-2 score and arm-3 outer evidence in one
uint8 score map.  A scalar modulo recovers the global arm length, and scalar
peak amplitudes make one fixed expansion kernel act as either a radius-2 or
radius-3 plus.  This removes the second score map, the selected score map, and
the dynamic 7x7 kernel tensor from the baseline.
"""

from __future__ import annotations

import csv
import pathlib
import subprocess
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = pathlib.Path(r"F:\kaggle")
BASELINE = (
    ROOT
    / "neurogolf-2026"
    / "work"
    / "e_high737801_recovered7_v1"
    / "models"
    / "task118.onnx"
)
SCORE_SCRIPT = ROOT / "neurogolf-2026" / "src" / "score_onnx.py"
WORKPLACE = ROOT / "NGC-work" / "workplace E"
OUTPUT_DIR = WORKPLACE / "optimized_onnx" / "task118_deep_20260713"
REPORT_PREFIX = WORKPLACE / "e_task118_deep_20260713"


def copy_value_info(value: onnx.ValueInfoProto) -> onnx.ValueInfoProto:
    result = onnx.ValueInfoProto()
    result.CopyFrom(value)
    return result


def initializer_map(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {
        initializer.name: numpy_helper.to_array(initializer).copy()
        for initializer in model.graph.initializer
    }


def plus_kernel(base: int, outer: int) -> np.ndarray:
    """Encode inner evidence in multiples of base and outer evidence below it."""
    kernel = np.zeros((1, 1, 7, 7), dtype=np.uint8)
    kernel[0, 0, 3, 3] = base
    for distance in (1, 2):
        kernel[0, 0, 3 - distance, 3] = 2 * base
        kernel[0, 0, 3 + distance, 3] = 2 * base
        kernel[0, 0, 3, 3 - distance] = 2 * base
        kernel[0, 0, 3, 3 + distance] = 2 * base
    kernel[0, 0, 0, 3] = outer
    kernel[0, 0, 6, 3] = outer
    kernel[0, 0, 3, 0] = outer
    kernel[0, 0, 3, 6] = outer
    return kernel


def expansion_kernel() -> np.ndarray:
    """Expand low peaks by radius 2 and high peaks by radius 3 after rounding."""
    kernel = np.zeros((1, 1, 7, 7), dtype=np.uint8)
    kernel[0, 0, 3, 3] = 255
    for distance in (1, 2):
        kernel[0, 0, 3 - distance, 3] = 255
        kernel[0, 0, 3 + distance, 3] = 255
        kernel[0, 0, 3, 3 - distance] = 255
        kernel[0, 0, 3, 3 + distance] = 255
    kernel[0, 0, 0, 3] = 1
    kernel[0, 0, 6, 3] = 1
    kernel[0, 0, 3, 0] = 1
    kernel[0, 0, 3, 6] = 1
    return kernel


def make_coded_model(
    baseline: onnx.ModelProto,
    *,
    base: int,
    outer: int,
    global_max_only: bool = False,
    suppress_zero_peaks: bool = False,
) -> onnx.ModelProto:
    arrays = initializer_map(baseline)
    initializers = [
        numpy_helper.from_array(arrays[name], name)
        for name in ("qscale", "qzp", "onehot8", "sl_s", "sl_e", "pads_hw", "pad_axes")
    ]
    initializers.extend(
        [
            numpy_helper.from_array(plus_kernel(base, outer), "scoreK"),
            numpy_helper.from_array(expansion_kernel(), "expandK"),
            numpy_helper.from_array(np.array(base, dtype=np.uint8), "arm_base"),
            numpy_helper.from_array(np.array(1, dtype=np.uint8), "peak_low"),
            numpy_helper.from_array(np.array(255, dtype=np.uint8), "peak_high"),
        ]
    )

    qconv_inputs = ["qscale", "qzp"]
    nodes = [
        helper.make_node("Slice", ["input", "sl_s", "sl_e"], ["ch2"]),
        helper.make_node("QuantizeLinear", ["ch2", "qscale", "qzp"], ["ch2_u8"]),
        helper.make_node(
            "QLinearConv",
            ["ch2_u8", *qconv_inputs, "scoreK", *qconv_inputs, *qconv_inputs],
            ["score"],
            pads=[1, 2, 1, 0],
            strides=[1, 1],
        ),
        helper.make_node("ReduceMax", ["score"], ["maxscore"], keepdims=0),
        helper.make_node("Mod", ["maxscore", "arm_base"], ["remainder"], fmod=0),
        helper.make_node("Equal", ["remainder", "qzp"], ["useArm2"]),
    ]
    if global_max_only:
        nodes.append(helper.make_node("Equal", ["score", "maxscore"], ["is_max"]))
    else:
        nodes.append(
            helper.make_node(
                "MaxPool",
                ["score"],
                ["pool"],
                kernel_shape=[11, 11],
                pads=[5, 5, 5, 5],
                strides=[1, 1],
            )
        )
        if suppress_zero_peaks:
            nodes.extend(
                [
                    helper.make_node("Sub", ["pool", "peak_low"], ["pool_minus_one"]),
                    helper.make_node("Greater", ["score", "pool_minus_one"], ["is_max"]),
                ]
            )
        else:
            nodes.append(helper.make_node("Equal", ["score", "pool"], ["is_max"]))
    nodes.extend(
        [
            helper.make_node("Where", ["useArm2", "peak_low", "peak_high"], ["peak"]),
            helper.make_node("Where", ["is_max", "peak", "qzp"], ["cand"]),
            helper.make_node(
                "QLinearConv",
                ["cand", *qconv_inputs, "expandK", *qconv_inputs, *qconv_inputs],
                ["expanded"],
                pads=[5, 4, 5, 6],
                strides=[1, 1],
            ),
            helper.make_node("Less", ["ch2_u8", "expanded"], ["mask_crop"]),
            helper.make_node(
                "Pad",
                ["mask_crop", "pads_hw", "", "pad_axes"],
                ["mask_b"],
                mode="constant",
            ),
            helper.make_node("Where", ["mask_b", "onehot8", "input"], ["output"]),
        ]
    )

    graph = helper.make_graph(
        nodes,
        f"task118_coded_b{base}_e{outer}",
        [copy_value_info(baseline.graph.input[0])],
        [copy_value_info(baseline.graph.output[0])],
        initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="NGC-workplace-E",
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=baseline.ir_version,
    )
    onnx.checker.check_model(model, full_check=True)
    return model


def make_fixed_score_model(baseline: onnx.ModelProto, score_name: str) -> onnx.ModelProto:
    """Remove the selected-score tensor while retaining baseline arm selection."""
    model = onnx.ModelProto()
    model.CopyFrom(baseline)
    kept: list[onnx.NodeProto] = []
    for node in model.graph.node:
        if list(node.output) == ["score"]:
            continue
        copied = onnx.NodeProto()
        copied.CopyFrom(node)
        for index, name in enumerate(copied.input):
            if name == "score":
                copied.input[index] = score_name
        kept.append(copied)
    del model.graph.node[:]
    model.graph.node.extend(kept)
    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    return model


def make_classify_then_selected_model(baseline: onnx.ModelProto) -> onnx.ModelProto:
    """Classify the arm length exactly, then compute only the selected score map.

    With the four outer weights reduced from 20 to 19, class_score is
    ``10 * s3_units - outer_red_count``.  If max(s2) == max(s3), a maximizing
    point with zero outer evidence exists and the maximum is divisible by 10.
    If max(s3) > max(s2), every s3 maximizer has outer evidence and the residue
    is non-zero.  The selected baseline kernel then reproduces the original
    center ordering without materializing both s2 and s3.
    """
    arrays = initializer_map(baseline)
    class_kernel = arrays["scoreK3"].copy()
    class_kernel[0, 0, 0, 3] = 19
    class_kernel[0, 0, 6, 3] = 19
    class_kernel[0, 0, 3, 0] = 19
    class_kernel[0, 0, 3, 6] = 19

    initializers = [
        numpy_helper.from_array(arrays[name], name)
        for name in (
            "qscale",
            "qzp",
            "scoreK2",
            "scoreK3",
            "onehot8",
            "sl_s",
            "sl_e",
            "pads_hw",
            "pad_axes",
        )
    ]
    initializers.extend(
        [
            numpy_helper.from_array(class_kernel, "classK"),
            numpy_helper.from_array(np.array(10, dtype=np.uint8), "class_base"),
        ]
    )

    qconv_inputs = ["qscale", "qzp"]
    nodes = [
        helper.make_node("Slice", ["input", "sl_s", "sl_e"], ["ch2"]),
        helper.make_node("QuantizeLinear", ["ch2", "qscale", "qzp"], ["ch2_u8"]),
        helper.make_node(
            "QLinearConv",
            ["ch2_u8", *qconv_inputs, "classK", *qconv_inputs, *qconv_inputs],
            ["class_score"],
            pads=[1, 2, 1, 0],
            strides=[1, 1],
        ),
        helper.make_node("ReduceMax", ["class_score"], ["class_max"], keepdims=0),
        helper.make_node("Mod", ["class_max", "class_base"], ["class_remainder"], fmod=0),
        helper.make_node("Equal", ["class_remainder", "qzp"], ["useArm2"]),
        helper.make_node("Where", ["useArm2", "scoreK2", "scoreK3"], ["selK"]),
        helper.make_node(
            "QLinearConv",
            ["ch2_u8", *qconv_inputs, "selK", *qconv_inputs, *qconv_inputs],
            ["score"],
            pads=[1, 2, 1, 0],
            strides=[1, 1],
        ),
        helper.make_node(
            "MaxPool",
            ["score"],
            ["pool"],
            kernel_shape=[11, 11],
            pads=[5, 5, 5, 5],
            strides=[1, 1],
        ),
        helper.make_node("Equal", ["score", "pool"], ["is_max"]),
        helper.make_node("Where", ["is_max", "score", "qzp"], ["cand"]),
        helper.make_node(
            "QLinearConv",
            ["cand", *qconv_inputs, "selK", *qconv_inputs, *qconv_inputs],
            ["expanded"],
            pads=[5, 4, 5, 6],
            strides=[1, 1],
        ),
        helper.make_node("Less", ["ch2_u8", "expanded"], ["mask_crop"]),
        helper.make_node(
            "Pad",
            ["mask_crop", "pads_hw", "", "pad_axes"],
            ["mask_b"],
            mode="constant",
        ),
        helper.make_node("Where", ["mask_b", "onehot8", "input"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes,
        "task118_classify_then_selected",
        [copy_value_info(baseline.graph.input[0])],
        [copy_value_info(baseline.graph.output[0])],
        initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="NGC-workplace-E",
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=baseline.ir_version,
    )
    onnx.checker.check_model(model, full_check=True)
    return model


def make_pooled_peak_model(
    baseline: onnx.ModelProto,
    *,
    stride: int,
    pooled_nms: bool,
) -> onnx.ModelProto:
    """Replace the full-resolution peak mask with pooled indices and MaxUnpool."""
    model = make_classify_then_selected_model(baseline)
    model.graph.initializer.append(
        numpy_helper.from_array(np.array([1, 1, 21, 23], dtype=np.int64), "unpool_shape")
    )
    replacement: list[onnx.NodeProto] = [
        helper.make_node(
            "MaxPool",
            ["score"],
            ["pooled", "pooled_indices"],
            kernel_shape=[11, 11],
            pads=[5, 5, 5, 5],
            strides=[stride, stride],
        )
    ]
    unpool_input = "pooled"
    if pooled_nms:
        replacement.extend(
            [
                helper.make_node(
                    "MaxPool",
                    ["pooled"],
                    ["pooled_neighbor_max"],
                    kernel_shape=[3, 3],
                    pads=[1, 1, 1, 1],
                    strides=[1, 1],
                ),
                helper.make_node(
                    "Equal",
                    ["pooled", "pooled_neighbor_max"],
                    ["pooled_is_max"],
                ),
                helper.make_node(
                    "Where",
                    ["pooled_is_max", "pooled", "qzp"],
                    ["pooled_cand"],
                ),
            ]
        )
        unpool_input = "pooled_cand"
    replacement.append(
        helper.make_node(
            "MaxUnpool",
            [unpool_input, "pooled_indices", "unpool_shape"],
            ["cand"],
            kernel_shape=[11, 11],
            pads=[5, 5, 5, 5],
            strides=[stride, stride],
        )
    )

    nodes: list[onnx.NodeProto] = []
    inserted = False
    for node in model.graph.node:
        output = node.output[0]
        if output == "pool":
            nodes.extend(replacement)
            inserted = True
        elif output in {"is_max", "cand"}:
            continue
        else:
            copied = onnx.NodeProto()
            copied.CopyFrom(node)
            nodes.append(copied)
    if not inserted:
        raise RuntimeError("baseline peak chain was not found")
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.value_info[:]
    model.graph.name = f"task118_pooled_peak_s{stride}_{'nms' if pooled_nms else 'raw'}"
    onnx.checker.check_model(model, full_check=True)
    return model


def save_model(model: onnx.ModelProto, variant: str) -> pathlib.Path:
    target_dir = OUTPUT_DIR / variant
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "task118.onnx"
    onnx.save(model, target)
    return target


def score(path: pathlib.Path, label: str) -> dict[str, str]:
    report = pathlib.Path(f"{REPORT_PREFIX}_{label}.csv")
    trace_dir = OUTPUT_DIR / label / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(SCORE_SCRIPT),
            str(path),
            "--output",
            str(report),
            "--trace-dir",
            str(trace_dir),
        ],
        check=True,
    )
    with report.open(newline="", encoding="utf-8") as handle:
        return next(csv.DictReader(handle))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = onnx.load(BASELINE)
    variants: list[tuple[str, pathlib.Path]] = []

    for base, outer in ((7, 3), (9, 4), (11, 5), (13, 6)):
        label = f"coded_b{base}_e{outer}"
        variants.append((label, save_model(make_coded_model(baseline, base=base, outer=outer), label)))

    for base, outer in ((7, 3), (9, 4), (11, 5), (13, 6)):
        label = f"coded_b{base}_e{outer}_zero_suppressed"
        variants.append(
            (
                label,
                save_model(
                    make_coded_model(
                        baseline,
                        base=base,
                        outer=outer,
                        suppress_zero_peaks=True,
                    ),
                    label,
                ),
            )
        )

    # Preserve the baseline s3 ordering exactly at the coarse scale and use
    # only the low residue to mark radius-3 outer evidence.
    for base in (5, 7, 9, 11):
        outer = 2 * base + 1
        label = f"ordered_b{base}_e{outer}_zero_suppressed"
        variants.append(
            (
                label,
                save_model(
                    make_coded_model(
                        baseline,
                        base=base,
                        outer=outer,
                        suppress_zero_peaks=True,
                    ),
                    label,
                ),
            )
        )

    label = "coded_b13_e6_global_max"
    variants.append(
        (
            label,
            save_model(
                make_coded_model(baseline, base=13, outer=6, global_max_only=True),
                label,
            ),
        )
    )
    for score_name in ("s2", "s3"):
        label = f"baseline_{score_name}_only"
        variants.append((label, save_model(make_fixed_score_model(baseline, score_name), label)))

    label = "classify_then_selected"
    variants.append(
        (
            label,
            save_model(make_classify_then_selected_model(baseline), label),
        )
    )

    for stride in (4, 5, 6):
        for pooled_nms in (False, True):
            label = f"pooled_peak_s{stride}_{'nms3' if pooled_nms else 'raw'}"
            variants.append(
                (
                    label,
                    save_model(
                        make_pooled_peak_model(
                            baseline,
                            stride=stride,
                            pooled_nms=pooled_nms,
                        ),
                        label,
                    ),
                )
            )

    rows: list[dict[str, str]] = []
    baseline_row = score(BASELINE, "baseline")
    baseline_row["variant"] = "baseline"
    rows.append(baseline_row)
    for label, path in variants:
        row = score(path, label)
        row["variant"] = label
        rows.append(row)

    summary = pathlib.Path(f"{REPORT_PREFIX}_summary.csv")
    fields = ["variant", *[key for key in rows[0] if key != "variant"]]
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['variant']}: status={row['status']} "
            f"pass={row['arc_agi_pass']}+{row['arc_gen_pass']} "
            f"cost={row['cost']}"
        )
    print(f"summary={summary}")


if __name__ == "__main__":
    main()
