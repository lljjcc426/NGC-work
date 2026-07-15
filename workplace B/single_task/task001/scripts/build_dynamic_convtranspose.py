from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = REPO_ROOT / "workplace C" / "neurogolf-2026-work" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from c_score_common import score_onnx  # noqa: E402


def initializer(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(value, name=name)


def build(output_path: Path) -> Path:
    # The input is a padded one-hot [1, 10, 30, 30] tensor. Its top-left 3x3
    # crop is also a valid ConvTranspose weight tensor [Cin=1, Cout=10, 30, 30].
    # A foreground-only 3x3 activation stamps that dynamic kernel at stride 3,
    # which is exactly the task's Kronecker product ordering.
    starts = initializer("starts", np.asarray([0, 1, 0, 0], dtype=np.int64))
    ends = initializer("ends", np.asarray([1, 10, 3, 3], dtype=np.int64))
    nodes = [
        helper.make_node(
            "Slice",
            ["input", "starts", "ends"],
            ["foreground_channels"],
            name="foreground_crop",
        ),
        helper.make_node(
            "ReduceMax",
            ["foreground_channels"],
            ["foreground_mask"],
            axes=[1],
            keepdims=1,
            name="foreground_mask",
        ),
        helper.make_node(
            "ConvTranspose",
            ["foreground_mask", "input"],
            ["output"],
            name="kronecker_stamp",
            strides=[3, 3],
            pads=[0, 0, 6, 6],
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "task001_dynamic_convtranspose",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        [starts, ends],
        value_info=[
            helper.make_tensor_value_info("foreground_channels", TensorProto.FLOAT, [1, 9, 3, 3]),
            helper.make_tensor_value_info("foreground_mask", TensorProto.FLOAT, [1, 1, 3, 3]),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="ngc-task001-dynamic-convtranspose",
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=10,
    )
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    reloaded = onnx.load(output_path)
    onnx.checker.check_model(reloaded, full_check=True)
    onnx.shape_inference.infer_shapes(reloaded, strict_mode=True)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "workplace B" / "single_task" / "task001" / "debug" / "task001_dynamic_convtranspose.onnx",
    )
    args = parser.parse_args()
    path = build(args.output)
    result = score_onnx("task001", path)
    print(result)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
