from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build(output_path: Path, geometry: str) -> Path:
    crop = np.zeros((1, 10, 2, 2), dtype=np.float32)
    crop[0, 0, 0, 0] = -1.0
    crop[0, 1, 0, 0] = 1.0

    recurrence = np.array([5, -1, -1, 2, 3, 1, 1, -2], dtype=np.float32)
    weights = np.zeros((1, 10, 8, 1), dtype=np.float32)
    weights[0, 0, :, 0] = -recurrence
    weights[0, 2, :, 0] = recurrence

    attrs: dict[str, object] = {"strides": [1, 1]}
    if geometry == "output-shape":
        attrs["output_shape"] = [30, 30]
    elif geometry == "negative-pads":
        attrs["pads"] = [0, 0, -17, -27]
    else:
        raise ValueError(geometry)

    nodes = [
        helper.make_node(
            "Conv",
            ["input", "bipolar_crop_weights"],
            ["bipolar6x3"],
            name="task003_bipolar_crop",
            kernel_shape=[2, 2],
            dilations=[24, 27],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
        ),
        helper.make_node(
            "ConvTranspose",
            ["bipolar6x3", "recurrence_weights"],
            ["output"],
            name="task003_hard_margin_recurrence",
            **attrs,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        f"task003_hard_margin_recurrence_{geometry}",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        [
            numpy_helper.from_array(crop, name="bipolar_crop_weights"),
            numpy_helper.from_array(weights, name="recurrence_weights"),
        ],
        value_info=[
            helper.make_tensor_value_info("bipolar6x3", TensorProto.FLOAT, [1, 1, 6, 3])
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="ngc-task003-hard-margin-recurrence",
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=10,
    )
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    reloaded = onnx.load(output_path)
    onnx.checker.check_model(reloaded, full_check=True)
    onnx.shape_inference.infer_shapes(reloaded, strict_mode=True, data_prop=True)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build task003's two-node hard-margin periodic recurrence.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometry", choices=["output-shape", "negative-pads"], required=True)
    args = parser.parse_args()
    print(build(args.output, args.geometry))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
