from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build(output_path: Path) -> Path:
    """Build a one-node classifier for support and same-color connectivity."""
    weights = np.zeros((10, 1, 5, 5), dtype=np.float32)
    bias = np.full(10, -10.0, dtype=np.float32)

    # The background channel separates the top-left-anchored valid rectangle
    # (including sparse cells to erase) from zero padding and retained motifs.
    # These coefficients are the hard-margin solution over all public examples.
    weights[0, 0] = np.array(
        [
            [2.83071143, -4.83540737, 2.03161932, -0.60170619, -0.61532441],
            [-0.81787587, -5.07850043, 11.62135086, -4.32887219, -2.52062299],
            [-1.47937701, 17.19433357, 32.55709478, 13.32456758, 2.22853565],
            [-2.46834155, -4.32887219, 12.98912108, -3.21389998, -3.18572435],
            [1.71918291, -3.18572435, 1.81748454, -0.85747828, 0.22383971],
        ],
        dtype=np.float32,
    )
    bias[0] = -16.72779213

    # Every foreground channel is positive exactly when its color occupies the
    # center and at least two orthogonal neighbors. Outer-ring weights stay 0.
    for color in range(1, 10):
        weights[color, 0, 2, 2] = 7.0
        weights[color, 0, 1, 2] = 2.0
        weights[color, 0, 2, 1] = 2.0
        weights[color, 0, 2, 3] = 2.0
        weights[color, 0, 3, 2] = 2.0

    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv",
                ["input", "W", "B"],
                ["output"],
                group=10,
                kernel_shape=[5, 5],
                pads=[2, 2, 2, 2],
            )
        ],
        "task193_depthwise_threshold",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        initializer=[
            numpy_helper.from_array(weights, name="W"),
            numpy_helper.from_array(bias, name="B"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "onnx" / "task193_candidate.onnx",
    )
    args = parser.parse_args()
    print(build(args.output))


if __name__ == "__main__":
    main()
