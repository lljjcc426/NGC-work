from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from rewrite_negative_pad_conv import rewrite_model


def make_model() -> onnx.ModelProto:
    weight = np.arange(1, 11, dtype=np.float32).reshape(1, 10, 1, 1)
    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv",
                ["input", "weight"],
                ["output"],
                kernel_shape=[1, 1],
                pads=[0, 0, -5, -7],
            )
        ],
        "negative_pad_crop",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 25, 23])],
        [numpy_helper.from_array(weight, "weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 10
    return model


def test_dilated_conv_matches_crop_reference() -> None:
    rng = np.random.default_rng(7)
    sample = rng.normal(size=(1, 10, 30, 30)).astype(np.float32)
    weights = np.arange(1, 11, dtype=np.float32).reshape(1, 10, 1, 1)
    reference = np.sum(sample[:, :, :25, :23] * weights, axis=1, keepdims=True)

    rewritten = rewrite_model(make_model())
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        rewritten.SerializeToString(), providers=["CPUExecutionProvider"], sess_options=options
    )
    actual = session.run(None, {"input": sample})[0]
    np.testing.assert_array_equal(actual, reference)

    conv = rewritten.graph.node[0]
    attrs = {attr.name: helper.get_attribute_value(attr) for attr in conv.attribute}
    assert attrs["pads"] == [0, 0, 0, 0]
    assert attrs["dilations"] == [5, 7]
    assert list(next(item for item in rewritten.graph.initializer if item.name == "weight").dims) == [1, 10, 2, 2]
