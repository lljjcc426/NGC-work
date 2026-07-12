from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


TASK_DIR = Path(__file__).resolve().parents[1]
SOURCE = TASK_DIR / "onnx" / "task158_candidate.onnx"
OUTPUT = TASK_DIR / "onnx" / "task158_pair3_candidate.onnx"


def array(model: onnx.ModelProto, name: str) -> np.ndarray:
    return next(numpy_helper.to_array(item) for item in model.graph.initializer if item.name == name)


def replace(model: onnx.ModelProto, name: str, value: np.ndarray) -> None:
    for index, item in enumerate(model.graph.initializer):
        if item.name == name:
            model.graph.initializer[index].CopyFrom(numpy_helper.from_array(value, name=name))
            return
    model.graph.initializer.append(numpy_helper.from_array(value, name=name))


def merged_pair_weights(weight: np.ndarray) -> np.ndarray:
    # Direction codes: d0=111, d1=100, d2=010, d3=001.
    return np.stack([weight[0] + weight[1], weight[0] + weight[2], weight[0] + weight[3]]).astype(np.int8)


def build(output_path: Path = OUTPUT) -> Path:
    model = onnx.load(str(SOURCE))
    graph = model.graph

    pair_biases = {1: -1, 2: -5, 3: -13}
    pair_scales = {1: 1.0, 2: 3.0, 3: 5.0}
    for scale in (1, 2, 3):
        replace(model, f"w_pair{scale}", merged_pair_weights(array(model, f"w_pair{scale}")))
        replace(model, f"b_pair{scale}", np.full(3, pair_biases[scale], dtype=np.int32))
        scale_name = f"pair_y_scale{scale}"
        replace(model, scale_name, np.array(pair_scales[scale], dtype=np.float32))
        node = next(item for item in graph.node if item.output == [f"pair_u8{scale}"])
        node.input[6] = scale_name

    # For stamp bits y0..y3, these three affine rows produce signed kernels
    # whose dot product with the direction codes is positive exactly where the
    # corresponding original orientation stamp is one.
    transform = np.array(
        [
            [1, 3, -1, -1],
            [1, -1, 3, -1],
            [1, -1, -1, 3],
        ],
        dtype=np.int8,
    ).reshape(3, 4, 1, 1)
    replace(model, "stamp_pair3_transform", transform)
    replace(model, "stamp_pair3_bias", np.full(3, -1, dtype=np.int32))

    stamp1_index = next(index for index, item in enumerate(graph.node) if item.output == ["stamp_w1"])
    transform_node = helper.make_node(
        "QLinearConv",
        [
            "stamp_w1",
            "q_xs",
            "q_wzp",
            "stamp_pair3_transform",
            "q_xs",
            "q_wzp",
            "q_xs",
            "q_wzp",
            "stamp_pair3_bias",
        ],
        ["stamp_pair3_w1"],
        name="stamp_pair3_transform_node",
        kernel_shape=[1, 1],
    )
    nodes = list(graph.node)
    nodes.insert(stamp1_index + 1, transform_node)
    for node in nodes:
        if node.output == ["fill_u81"]:
            node.input[3] = "stamp_pair3_w1"
        elif node.output == ["stamp_w2"]:
            node.input[0] = "stamp_pair3_w1"
        elif node.output == ["stamp_w3"]:
            node.input[0] = "stamp_pair3_w1"
    del graph.node[:]
    graph.node.extend(nodes)

    replace(model, "stamp_size2", np.array([1, 3, 6, 6], dtype=np.int64))
    replace(model, "stamp_size3", np.array([1, 3, 9, 9], dtype=np.int64))

    # Keep profiling metadata so the official scorer can account for memory,
    # but update the tensors affected by the exact 4->3 channel rewrite.
    three_channel_names = {
        "stamp_w2",
        "stamp_w3",
        "pair_u81",
        "pair_u82",
        "pair_u83",
    }
    for value_info in graph.value_info:
        if value_info.name in three_channel_names:
            value_info.type.tensor_type.shape.dim[1].dim_value = 3
    graph.value_info.append(
        helper.make_tensor_value_info("stamp_pair3_w1", onnx.TensorProto.INT8, [1, 3, 3, 3])
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(output_path))
    return output_path


if __name__ == "__main__":
    print(build())
