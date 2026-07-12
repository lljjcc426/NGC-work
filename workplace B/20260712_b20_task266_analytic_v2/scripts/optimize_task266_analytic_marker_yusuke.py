from __future__ import annotations

from collections import Counter

import numpy as np
import onnx
from onnx import helper, numpy_helper


def _init(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(value, name=name)


def transform(model: onnx.ModelProto) -> tuple[onnx.ModelProto, Counter]:
    current = onnx.ModelProto.FromString(model.SerializeToString())
    if [node.op_type for node in current.graph.node] != ["Conv", "Relu", "Conv"]:
        return current, Counter()
    if tuple(current.graph.initializer[0].dims) != (1, 10, 3, 3):
        return current, Counter()

    encode = np.ones((1, 10, 1, 1), dtype=np.float32)
    encode[0, 2, 0, 0] = 2.0

    route = np.zeros((10, 1, 3, 3), dtype=np.float32)
    route[0, 0] = np.asarray(
        [[-4, 6, -4], [6, 2, 6], [-4, 6, -4]], dtype=np.float32
    )
    route[3, 0] = np.asarray(
        [[0, 0, 0], [0, 0, 0], [0, -2, 2]], dtype=np.float32
    )
    route[6, 0] = np.asarray(
        [[0, 0, 0], [-3, 1, 0], [3, 1, 0]], dtype=np.float32
    )
    route[7, 0] = np.asarray(
        [[3, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.float32
    )
    route[8, 0] = np.asarray(
        [[-1, -2, 3], [1, 1, 1], [0, 0, 0]], dtype=np.float32
    )

    bias = np.full(10, -1.0, dtype=np.float32)
    bias[0] = -9.0
    bias[3] = -1.0
    bias[6] = -4.0
    bias[7] = -8.0
    bias[8] = -5.0

    del current.graph.initializer[:]
    current.graph.initializer.extend(
        [
            _init("task266_encode", encode),
            _init("task266_route", route),
            _init("task266_bias", bias),
        ]
    )
    del current.graph.node[:]
    current.graph.node.extend(
        [
            helper.make_node(
                "Conv",
                ["input", "task266_encode"],
                ["task266_code"],
                kernel_shape=[1, 1],
                pads=[0, 0, -27, -25],
                name="task266_code",
            ),
            helper.make_node(
                "Conv",
                ["task266_code", "task266_route", "task266_bias"],
                ["output"],
                kernel_shape=[3, 3],
                pads=[1, 1, 28, 26],
                name="output",
            ),
        ]
    )
    del current.graph.value_info[:]

    onnx.checker.check_model(current, full_check=True)
    onnx.shape_inference.infer_shapes(current, strict_mode=True)
    return current, Counter({"task266_analytic_marker": 1})
