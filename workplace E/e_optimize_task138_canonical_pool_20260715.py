"""Build a one-pool canonical-direction rewrite for task138."""

from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper


REPO = pathlib.Path(__file__).resolve().parents[1]
WORKSPACE = pathlib.Path(__file__).resolve().parent
NGC_ROOT = REPO.parent / "neurogolf-2026"
C_SCRIPTS = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"

os.environ.setdefault("NEUROGOLF_DATA_ROOT", str(NGC_ROOT / "data"))
os.environ.setdefault(
    "NEUROGOLF_UTILS_PATH",
    str(NGC_ROOT / "data" / "neurogolf_utils" / "neurogolf_utils.py"),
)
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(C_SCRIPTS))

import e_optimize_task138_deep_20260713 as legacy  # noqa: E402
from c_score_common import score_onnx  # noqa: E402


DEFAULT_PARENT = (
    NGC_ROOT
    / "work"
    / "e_round_738588_isolate_20260715_v1"
    / "task118"
    / "package"
    / "models"
    / "task138.onnx"
)
DEFAULT_OUTPUT = (
    WORKSPACE
    / "optimized_onnx"
    / "task138_canonical_pool_20260715"
    / "task138.onnx"
)
DEFAULT_REPORT = WORKSPACE / "e_task138_canonical_pool_20260715.json"


def build(base: onnx.ModelProto) -> onnx.ModelProto:
    model = copy.deepcopy(base)
    original = list(model.graph.node)
    expected = [
        "Conv",
        "Cast",
        "Conv",
        "Conv",
        "ArgMax",
        "Cast",
        "Squeeze",
        "ArgMax",
        "Cast",
        "Squeeze",
        "ArgMax",
        "Cast",
        "Squeeze",
        "ArgMax",
        "Cast",
        "Squeeze",
        "Sub",
        "Sub",
    ]
    if [node.op_type for node in original[:18]] != expected:
        raise ValueError("Unexpected task138 parent graph prefix")

    nodes = list(original[:35])
    nodes.extend(
        [
            helper.make_node(
                "Sub", ["i32_twenty_two", "row_span_minus_one"], ["square_pad_h_i32"]
            ),
            helper.make_node(
                "Sub", ["i32_twenty_two", "col_span_minus_one"], ["square_pad_w_i32"]
            ),
            helper.make_node(
                "Cast", ["square_pad_h_i32"], ["square_pad_h_i64"], to=TensorProto.INT64
            ),
            helper.make_node(
                "Cast", ["square_pad_w_i32"], ["square_pad_w_i64"], to=TensorProto.INT64
            ),
            helper.make_node(
                "Concat",
                ["pad_prefix_six_i64", "square_pad_h_i64", "square_pad_w_i64"],
                ["square_pads_i64"],
                axis=0,
            ),
            helper.make_node(
                "Pad", ["inner_color", "square_pads_i64", "u8_zero"], ["inner_square"]
            ),
            helper.make_node(
                "Gather", ["inner_square", "reverse21_i64"], ["inner_square_rev"], axis=2
            ),
            helper.make_node(
                "Transpose", ["inner_square"], ["inner_square_t"], perm=[0, 1, 3, 2]
            ),
            helper.make_node(
                "Gather", ["inner_square_t", "reverse21_i64"], ["inner_square_t_rev"], axis=2
            ),
            helper.make_node(
                "Where",
                ["top_match_4", "inner_square", "inner_square_rev"],
                ["vertical_canonical"],
            ),
            helper.make_node(
                "Where",
                ["left_match_4", "inner_square_t", "inner_square_t_rev"],
                ["horizontal_canonical"],
            ),
            helper.make_node(
                "Where",
                ["horizontal_flag_4", "horizontal_canonical", "vertical_canonical"],
                ["canonical_seed"],
            ),
            helper.make_node(
                "MaxPool",
                ["canonical_seed"],
                ["canonical_fill"],
                kernel_shape=[22, 1],
                pads=[0, 0, 21, 0],
            ),
            helper.make_node(
                "Gather", ["canonical_fill", "reverse21_i64"], ["canonical_fill_rev"], axis=2
            ),
            helper.make_node(
                "Where",
                ["top_match_4", "canonical_fill", "canonical_fill_rev"],
                ["vertical_fill_square"],
            ),
            helper.make_node(
                "Where",
                ["left_match_4", "canonical_fill", "canonical_fill_rev"],
                ["horizontal_fill_axis"],
            ),
            helper.make_node(
                "Transpose",
                ["horizontal_fill_axis"],
                ["horizontal_fill_square"],
                perm=[0, 1, 3, 2],
            ),
            helper.make_node(
                "Where",
                ["horizontal_flag_4", "horizontal_fill_square", "vertical_fill_square"],
                ["fill_square"],
            ),
            helper.make_node(
                "Sub", ["row_span_minus_one", "i32_one"], ["inner_h_i32"]
            ),
            helper.make_node(
                "Sub", ["col_span_minus_one", "i32_one"], ["inner_w_i32"]
            ),
            helper.make_node(
                "Concat", ["inner_h_i32", "inner_w_i32"], ["fill_inner_ends_i32"], axis=0
            ),
            helper.make_node(
                "Slice",
                [
                    "fill_square",
                    "square_slice_starts_i32",
                    "fill_inner_ends_i32",
                    "crop_axes_i32",
                ],
                ["fill_inner"],
            ),
        ]
    )
    nodes.extend(original[42:])
    legacy.set_nodes(model, nodes)
    legacy.replace_initializers(
        model,
        [
            legacy.initializer("i32_twenty_two", np.asarray([22], dtype=np.int32)),
            legacy.initializer("reverse21_i64", np.arange(20, -1, -1, dtype=np.int64)),
            legacy.initializer("square_slice_starts_i32", np.asarray([0, 0], dtype=np.int32)),
        ],
    )

    legacy.annotate(model, "crop_color", TensorProto.UINT8, [1, 1, 22, 23])
    legacy.annotate(model, "inner_color", TensorProto.UINT8, [1, 1, 20, 21])
    legacy.annotate(model, "active_color_4", TensorProto.UINT8, [1, 1, 1, 1])
    legacy.annotate(model, "side_probe_row", TensorProto.UINT8, [1, 1, 1, 23])
    for name in (
        "left_color_4",
        "right_color_4",
    ):
        legacy.annotate(model, name, TensorProto.UINT8, [1, 1, 1, 1])
    for name in (
        "top_match_4",
        "left_match_4",
        "right_match_4",
        "horizontal_flag_4",
    ):
        legacy.annotate(model, name, TensorProto.BOOL, [1, 1, 1, 1])
    for name in (
        "inner_square",
        "inner_square_rev",
        "inner_square_t",
        "inner_square_t_rev",
        "vertical_canonical",
        "horizontal_canonical",
        "canonical_seed",
        "canonical_fill",
        "canonical_fill_rev",
        "vertical_fill_square",
        "horizontal_fill_axis",
        "horizontal_fill_square",
        "fill_square",
    ):
        legacy.annotate(model, name, TensorProto.UINT8, [1, 1, 21, 21])
    legacy.annotate(model, "fill_inner", TensorProto.UINT8, [1, 1, 20, 21])
    legacy.annotate(model, "fill_padded", TensorProto.UINT8, [1, 1, 22, 23])
    legacy.annotate(model, "color_grid_out", TensorProto.UINT8, [1, 1, 22, 23])
    legacy.annotate(model, "color_grid_clamped", TensorProto.UINT8, [1, 1, 30, 30])
    return legacy.finish(model)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=pathlib.Path, default=DEFAULT_PARENT)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=pathlib.Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    candidate = build(onnx.load(args.parent))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(candidate, args.output)

    parent_score = score_onnx("task138", args.parent, validate_all=True)
    candidate_score = score_onnx("task138", args.output, validate_all=True)
    report = {
        "task": "task138",
        "method": "canonicalize four directions, run one MaxPool, then invert",
        "parent": vars(parent_score),
        "candidate": vars(candidate_score),
        "accepted_local": bool(
            candidate_score.ok
            and candidate_score.examples_checked == candidate_score.examples_passed
            and candidate_score.cost is not None
            and parent_score.cost is not None
            and candidate_score.cost < parent_score.cost
        ),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["accepted_local"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
