from __future__ import annotations

from collections import Counter

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _init(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(value, name=name)


def transform(model: onnx.ModelProto) -> tuple[onnx.ModelProto, Counter]:
    current = onnx.ModelProto.FromString(model.SerializeToString())
    outputs = {name for node in current.graph.node for name in node.output}
    if not {"code32", "signed_pattern", "route_w", "output"}.issubset(outputs):
        return current, Counter()

    keep = next(node for node in current.graph.node if node.output[0] == "code32")
    keep = onnx.NodeProto.FromString(keep.SerializeToString())

    additions = [
        _init("task001_f32_zero", np.asarray(0, dtype=np.float32)),
        _init("task001_u8_scale", np.asarray(1, dtype=np.float32)),
        _init("task001_u8_zero", np.asarray(0, dtype=np.uint8)),
        _init("task001_i8_zero", np.asarray(0, dtype=np.int8)),
        _init("task001_left_shape", np.asarray([1, 1, 9, 1], dtype=np.int64)),
        _init("task001_right_shape", np.asarray([1, 1, 1, 9], dtype=np.int64)),
        _init("task001_fg_codes", np.arange(1, 10, dtype=np.float32).reshape(9, 1, 1, 1)),
        _init("task001_bg_weight", np.asarray([-1], dtype=np.int8).reshape(1, 1, 1, 1)),
        _init("task001_bias", np.asarray([1] + [0] * 9, dtype=np.int32)),
    ]

    nodes = [
        keep,
        helper.make_node(
            "Greater",
            ["code32", "task001_f32_zero"],
            ["task001_occupied_b"],
            name="task001_occupied_b",
        ),
        helper.make_node(
            "Cast",
            ["task001_occupied_b"],
            ["task001_occupied_u8"],
            to=TensorProto.UINT8,
            name="task001_occupied_u8",
        ),
        helper.make_node(
            "Reshape",
            ["task001_occupied_u8", "task001_left_shape"],
            ["task001_left"],
            name="task001_left",
        ),
        helper.make_node(
            "Reshape",
            ["task001_occupied_u8", "task001_right_shape"],
            ["task001_right"],
            name="task001_right",
        ),
        helper.make_node(
            "QLinearMatMul",
            [
                "task001_left",
                "task001_u8_scale",
                "task001_u8_zero",
                "task001_right",
                "task001_u8_scale",
                "task001_u8_zero",
                "task001_u8_scale",
                "task001_u8_zero",
            ],
            ["task001_pattern_u8"],
            name="task001_pattern_u8",
        ),
        helper.make_node(
            "ReduceMax",
            ["code32"],
            ["task001_color"],
            axes=[2, 3],
            keepdims=1,
            name="task001_color",
        ),
        helper.make_node(
            "Equal",
            ["task001_color", "task001_fg_codes"],
            ["task001_fg_match"],
            name="task001_fg_match",
        ),
        helper.make_node(
            "Cast",
            ["task001_fg_match"],
            ["task001_fg_weight"],
            to=TensorProto.INT8,
            name="task001_fg_weight",
        ),
        helper.make_node(
            "Concat",
            ["task001_bg_weight", "task001_fg_weight"],
            ["task001_route"],
            axis=0,
            name="task001_route",
        ),
        helper.make_node(
            "QLinearConv",
            [
                "task001_pattern_u8",
                "task001_u8_scale",
                "task001_u8_zero",
                "task001_route",
                "task001_u8_scale",
                "task001_i8_zero",
                "task001_u8_scale",
                "task001_u8_zero",
                "task001_bias",
            ],
            ["output"],
            kernel_shape=[1, 1],
            pads=[0, 0, 21, 21],
            name="output",
        ),
    ]

    del current.graph.node[:]
    current.graph.node.extend(nodes)
    del current.graph.initializer[:]
    current.graph.initializer.extend(
        [
            onnx.TensorProto.FromString(model.graph.initializer[0].SerializeToString()),
            *additions,
        ]
    )
    current.graph.output[0].type.tensor_type.elem_type = TensorProto.UINT8
    del current.graph.value_info[:]

    onnx.checker.check_model(current, full_check=True)
    onnx.shape_inference.infer_shapes(current, strict_mode=True)
    return current, Counter({"task001_qlinear_outer": 1})
