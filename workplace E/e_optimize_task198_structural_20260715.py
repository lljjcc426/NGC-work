from __future__ import annotations

import argparse
import copy
from pathlib import Path

import onnx
from onnx import helper


def node_by_output(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    return {
        output: node
        for node in model.graph.node
        for output in node.output
        if output
    }


def save_variant(
    source: onnx.ModelProto,
    nodes: list[onnx.NodeProto],
    path: Path,
) -> None:
    model = copy.deepcopy(source)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    print(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = onnx.load(args.source)
    by_output = node_by_output(source)

    common_prefix = [
        by_output[name]
        for name in (
            "labf",
            "lab",
            "row_cnt",
            "row_w0",
            "row_door",
            "col_door",
            "active_b",
            "active_sq",
        )
    ]
    common_suffix = [
        by_output[name]
        for name in (
            "room_active",
            "room_au8",
            "rb",
            "g1",
            "room_u8",
            "out_label",
            "output",
        )
    ]

    threshold_nodes = common_prefix + [
        by_output["act32"],
        helper.make_node(
            "ArgMax",
            ["act32"],
            ["fam_s"],
            axis=0,
            keepdims=0,
            name="select_family",
        ),
        by_output["room_score"],
        helper.make_node(
            "Gather",
            ["thrf", "fam_s"],
            ["thr_sel"],
            axis=0,
            name="select_threshold",
        ),
    ] + common_suffix
    save_variant(
        source,
        threshold_nodes,
        args.output_dir / "threshold_gather" / "task198.onnx",
    )

    selected_nodes = common_prefix + [
        helper.make_node(
            "ArgMax",
            ["active_sq"],
            ["fam_s"],
            axis=0,
            keepdims=0,
            name="select_family_u8",
        ),
        helper.make_node(
            "Gather",
            ["rm32", "fam_s"],
            ["rm_sel"],
            axis=0,
            name="select_room_matrix",
        ),
        helper.make_node(
            "Einsum",
            ["input", "e0c", "rm_sel", "rm_sel"],
            ["room_score"],
            equation="bkhw,kc,ih,jw->bcij",
            name="selected_room_score",
        ),
        helper.make_node(
            "Gather",
            ["thrf", "fam_s"],
            ["thr_sel"],
            axis=0,
            name="select_threshold",
        ),
    ] + common_suffix
    save_variant(
        source,
        selected_nodes,
        args.output_dir / "selected_family" / "task198.onnx",
    )


if __name__ == "__main__":
    main()
