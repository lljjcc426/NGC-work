from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def build(source: Path, output: Path) -> Path:
    model = deepcopy(onnx.load(source))
    nodes = list(model.graph.node)
    if not nodes or nodes[0].op_type != "Einsum" or nodes[0].output != ["templ_scalar_f"]:
        raise RuntimeError("unexpected task075 source graph")

    # The original coefficient tensor is exactly rank one in color and uses
    # two 3-of-30 selectors. A cropped 1x1 convolution performs both actions
    # without storing either selector.
    color_kernel = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    replacement = helper.make_node(
        "Conv",
        ["input", "template_color_kernel"],
        ["templ_scalar_f"],
        name="rank1_color_crop",
        kernel_shape=[1, 1],
        pads=[0, 0, -27, -27],
    )
    nodes[0].CopyFrom(replacement)

    marker = next(node for node in nodes if node.output == ["markers_f"])
    if marker.op_type != "Slice":
        raise RuntimeError("unexpected task075 marker extraction")
    marker.CopyFrom(
        helper.make_node(
            "Conv",
            ["input", "marker_channel_kernel"],
            ["markers_f"],
            name="marker_stride_crop",
            kernel_shape=[1, 1],
            pads=[-1, -5, -22, -18],
            strides=[3, 3],
        )
    )

    removed = {
        "color_weights",
        "sel3",
        "one_k",
        "axes3",
        "marker_starts",
        "marker_ends",
        "marker_steps",
    }
    kept = [item for item in model.graph.initializer if item.name not in removed]
    kept.append(numpy_helper.from_array(color_kernel, name="template_color_kernel"))
    marker_kernel = np.zeros((1, 10, 1, 1), dtype=np.float32)
    marker_kernel[0, 1, 0, 0] = 1.0
    kept.append(numpy_helper.from_array(marker_kernel, name="marker_channel_kernel"))
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    del model.graph.value_info[:]

    onnx.checker.check_model(model, full_check=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.source, args.output))


if __name__ == "__main__":
    main()
