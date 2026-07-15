from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


SHAPE = [1, 10, 30, 30]


def build() -> onnx.ModelProto:
    values = numpy_helper.from_array(np.ones(2, dtype=np.float32), "color_values")
    indices = numpy_helper.from_array(
        np.asarray([[0, 0], [2, 1]], dtype=np.int64), "color_indices"
    )
    color_map = helper.make_sparse_tensor(values, indices, [10, 10])
    tensor = lambda name: helper.make_tensor_value_info(
        name, TensorProto.FLOAT, SHAPE
    )
    node = helper.make_node(
        "Einsum",
        ["input", "color_map"],
        ["output"],
        name="output",
        equation="burc,ku->bkrc",
    )
    graph = helper.make_graph([node], "sparse_probe", [tensor("input")], [tensor("output")])
    graph.sparse_initializer.extend([color_map])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    model = build()
    output = Path(__file__).with_name("sparse_initializer_probe.onnx")
    onnx.save(model, output)
    session = ort.InferenceSession(model.SerializeToString())
    value = np.zeros(SHAPE, dtype=np.float32)
    value[0, 0, :6, :3] = 1
    value[0, 1, 0, 0] = 1
    value[0, 0, 0, 0] = 0
    result = session.run(["output"], {"input": value})[0]
    assert result[0, 2, 0, 0] == 1
    assert result[0, 0, 1, 1] == 1
    print(output)


if __name__ == "__main__":
    main()
