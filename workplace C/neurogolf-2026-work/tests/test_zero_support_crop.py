from __future__ import annotations

from copy import deepcopy

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from full400_zero_support_crop import crop_zero_support_detailed


def _conv_model(
    weight: np.ndarray,
    *,
    pads: tuple[int, int, int, int] = (1, 1, 1, 1),
    dilation: tuple[int, int] = (1, 1),
    auto_pad: str | None = None,
    shared: bool = False,
) -> onnx.ModelProto:
    nodes = [
        helper.make_node(
            "Conv",
            ["input", "weight"],
            ["output"],
            name="conv",
            pads=pads,
            dilations=dilation,
            **({"auto_pad": auto_pad} if auto_pad else {}),
        )
    ]
    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, weight.shape[0], 5, 5])]
    if shared:
        nodes.append(helper.make_node("Conv", ["input", "weight"], ["other"], name="other", pads=pads))
        outputs.append(helper.make_tensor_value_info("other", TensorProto.FLOAT, [1, weight.shape[0], 5, 5]))
    graph = helper.make_graph(
        nodes,
        "conv",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, weight.shape[1], 5, 5])],
        outputs,
        [numpy_helper.from_array(weight.astype(np.float32), name="weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)], ir_version=10)
    return model


def _qconv_model(weight: np.ndarray, zero_point: np.ndarray) -> onnx.ModelProto:
    initializers = [
        numpy_helper.from_array(np.asarray(1.0, dtype=np.float32), name="xs"),
        numpy_helper.from_array(np.asarray(0, dtype=np.uint8), name="xz"),
        numpy_helper.from_array(np.asarray(1.0, dtype=np.float32), name="ws"),
        numpy_helper.from_array(weight, name="weight"),
        numpy_helper.from_array(zero_point, name="wz"),
        numpy_helper.from_array(np.asarray(1.0, dtype=np.float32), name="ys"),
        numpy_helper.from_array(np.asarray(0, dtype=np.uint8), name="yz"),
    ]
    node = helper.make_node(
        "QLinearConv",
        ["input", "xs", "xz", "weight", "ws", "wz", "ys", "yz"],
        ["output"],
        name="qconv",
        pads=[1, 1, 1, 1],
    )
    graph = helper.make_graph(
        [node],
        "qconv",
        [helper.make_tensor_value_info("input", TensorProto.UINT8, [1, weight.shape[1], 5, 5])],
        [helper.make_tensor_value_info("output", TensorProto.UINT8, [1, weight.shape[0], 5, 5])],
        initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)], ir_version=10)


def test_conv_zero_border_is_cropped() -> None:
    weight = np.zeros((1, 1, 3, 3), dtype=np.float32)
    weight[:, :, 1:, 1:] = 1
    model = _conv_model(weight)
    changes, rejected = crop_zero_support_detailed(model)
    assert not rejected
    assert changes[0]["new_shape"] == [1, 1, 2, 2]
    assert changes[0]["new_pads"] == [0, 0, 1, 1]


def test_conv_without_zero_border_is_unchanged() -> None:
    changes, rejected = crop_zero_support_detailed(_conv_model(np.ones((1, 1, 3, 3), np.float32)))
    assert changes == []
    assert rejected == []


def test_negative_padding_is_rejected_without_mutation() -> None:
    weight = np.zeros((1, 1, 3, 3), dtype=np.float32)
    weight[:, :, 1:, :] = 1
    model = _conv_model(weight, pads=(0, 1, 1, 1))
    original = numpy_helper.to_array(model.graph.initializer[0]).copy()
    changes, rejected = crop_zero_support_detailed(model)
    assert changes == []
    assert rejected[0]["reason"] == "rejected_negative_padding"
    np.testing.assert_array_equal(numpy_helper.to_array(model.graph.initializer[0]), original)


def test_qlinear_zero_point_zero() -> None:
    weight = np.zeros((1, 1, 3, 3), dtype=np.uint8)
    weight[:, :, 1:, :] = 2
    changes, rejected = crop_zero_support_detailed(_qconv_model(weight, np.asarray(0, np.uint8)))
    assert not rejected
    assert changes[0]["new_shape"] == [1, 1, 2, 3]


def test_qlinear_zero_point_128() -> None:
    weight = np.full((1, 1, 3, 3), 128, dtype=np.uint8)
    weight[:, :, 1:, :] = 130
    changes, rejected = crop_zero_support_detailed(_qconv_model(weight, np.asarray(128, np.uint8)))
    assert not rejected
    assert changes[0]["weight_zero_point"] == [128]


def test_qlinear_per_output_channel_zero_point() -> None:
    weight = np.empty((2, 1, 3, 3), dtype=np.uint8)
    weight[0].fill(127)
    weight[1].fill(129)
    weight[:, :, 1:, :] += 1
    zero = np.asarray([127, 129], dtype=np.uint8)
    changes, rejected = crop_zero_support_detailed(_qconv_model(weight, zero))
    assert not rejected
    assert changes[0]["new_shape"] == [2, 1, 2, 3]


def test_shared_weight_is_skipped() -> None:
    weight = np.zeros((1, 1, 3, 3), dtype=np.float32)
    weight[:, :, 1:, :] = 1
    changes, _ = crop_zero_support_detailed(_conv_model(weight, shared=True))
    assert changes == []


def test_auto_pad_is_skipped() -> None:
    weight = np.zeros((1, 1, 3, 3), dtype=np.float32)
    weight[:, :, 1:, :] = 1
    changes, _ = crop_zero_support_detailed(_conv_model(weight, auto_pad="SAME_UPPER"))
    assert changes == []


def test_dilation_adjusts_padding() -> None:
    weight = np.zeros((1, 1, 3, 3), dtype=np.float32)
    weight[:, :, 1:, :] = 1
    changes, rejected = crop_zero_support_detailed(
        _conv_model(weight, pads=(2, 2, 2, 2), dilation=(2, 2))
    )
    assert not rejected
    assert changes[0]["new_pads"] == [0, 2, 2, 2]


def test_conv_output_is_exact_after_crop() -> None:
    weight = np.zeros((1, 1, 3, 3), dtype=np.float32)
    weight[:, :, 1:, 1:] = np.asarray([[1, 2], [3, 4]], dtype=np.float32)
    original = _conv_model(weight)
    cropped = deepcopy(original)
    changes, rejected = crop_zero_support_detailed(cropped)
    assert changes and not rejected
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    left = ort.InferenceSession(original.SerializeToString(), options, providers=["CPUExecutionProvider"])
    right = ort.InferenceSession(cropped.SerializeToString(), options, providers=["CPUExecutionProvider"])
    value = np.arange(25, dtype=np.float32).reshape(1, 1, 5, 5)
    np.testing.assert_array_equal(left.run(None, {"input": value})[0], right.run(None, {"input": value})[0])
