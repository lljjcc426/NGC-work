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


def build(output_path: Path) -> Path:
    weights = np.zeros((1, 10, 2, 2), dtype=np.float32)
    weights[0, 1:, 0, 0] = 1.0
    nodes = [
        helper.make_node(
            "Conv",
            ["input", "foreground_weights"],
            ["foreground_mask"],
            name="extract_top_left_foreground_mask",
            kernel_shape=[2, 2],
            dilations=[27, 27],
            pads=[0, 0, 0, 0],
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
        "task001_dilated_dynamic_convtranspose",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        [numpy_helper.from_array(weights, "foreground_weights")],
        value_info=[
            helper.make_tensor_value_info("foreground_mask", TensorProto.FLOAT, [1, 1, 3, 3])
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="ngc-task001-dilated-dynamic-convtranspose",
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
        default=REPO_ROOT
        / "workplace B"
        / "single_task"
        / "task001"
        / "onnx"
        / "task001_candidate.onnx",
    )
    args = parser.parse_args()
    path = build(args.output)
    result = score_onnx("task001", path, validate_all=True)
    print(result)
    return 0 if result.ok and result.examples_passed == result.examples_checked else 1


if __name__ == "__main__":
    raise SystemExit(main())
