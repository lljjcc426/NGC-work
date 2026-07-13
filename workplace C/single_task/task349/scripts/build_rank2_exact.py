from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


TASK_ROOT = Path(__file__).resolve().parents[1]
SOURCE = TASK_ROOT / "onnx" / "task349_candidate.onnx"
SOLUTION = TASK_ROOT / "rank2_exact_solution.npz"


def replace_initializer(model: onnx.ModelProto, name: str, value: np.ndarray) -> None:
    tensor = numpy_helper.from_array(value, name=name)
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name == name:
            model.graph.initializer[index].CopyFrom(tensor)
            return
    model.graph.initializer.append(tensor)


def build(output: Path) -> Path:
    solution = np.load(SOLUTION)
    model = onnx.load(SOURCE)
    replace_initializer(model, "h_kernel_combined_i8", solution["detector"].astype(np.int8))
    replace_initializer(model, "h_bias_combined_i32", solution["detector_bias"].astype(np.int32))
    replace_initializer(model, "halo_weight_i8", solution["halo"].astype(np.int8))
    replace_initializer(model, "halo_bias_i32", solution["halo_bias"].astype(np.int32))
    model.graph.name = "task349_exact_rank2_width_halo"
    model.doc_string = (
        "Two-channel exact integer encoding: widths 6/8/10 use amplitudes "
        "1/8/127 and widths 2/4 use amplitudes 1/16."
    )
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)
    onnx.checker.check_model(onnx.load(output), full_check=True)
    return output


if __name__ == "__main__":
    print(build(TASK_ROOT / "onnx" / "task349_rank2_exact.onnx"))
