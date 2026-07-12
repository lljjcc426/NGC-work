from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


TASK_ROOT = Path(__file__).resolve().parents[1]
SOURCE = TASK_ROOT / "onnx" / "task349_candidate.onnx"


def replace_initializer(model: onnx.ModelProto, name: str, value: np.ndarray) -> None:
    tensor = numpy_helper.from_array(value, name=name)
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name == name:
            model.graph.initializer[index].CopyFrom(tensor)
            return
    model.graph.initializer.append(tensor)


def build(output: Path) -> Path:
    """Encode widths 2/4 and widths 6/8/10 in two collision-aware channels."""
    model = onnx.load(str(SOURCE))
    old_halo = next(
        numpy_helper.to_array(value)
        for value in model.graph.initializer
        if value.name == "halo_weight_i8"
    )
    detector = np.zeros((2, 1, 2, 11), dtype=np.int8)
    detector[0, 0, 1] = [-127, 0, 127, -1, 127, -127, 0, 0, 0, 0, 0]
    detector[1, 0, 1] = [-127, 127, 1, 127, -127, 127, 127, 126, -125, -2, 127]
    detector_bias = np.array([-126, -381], dtype=np.int32)

    halo = np.zeros((1, 2, 11, 20), dtype=np.int8)
    small_0 = old_halo[0, 0] > 0
    large_0 = old_halo[0, 1] > 0
    halo[0, 0, large_0 & ~small_0] = 1
    halo[0, 0, small_0] = 127

    support_6 = old_halo[0, 2] > 0
    support_8 = old_halo[0, 3] > 0
    support_10 = old_halo[0, 4] > 0
    halo[0, 1, support_10 & ~support_8] = 1
    halo[0, 1, support_8 & ~support_6] = 64
    halo[0, 1, support_6] = 127

    replace_initializer(model, "h_kernel_combined_i8", detector)
    replace_initializer(model, "h_bias_combined_i32", detector_bias)
    replace_initializer(model, "halo_weight_i8", halo)
    replace_initializer(model, "halo_bias_i32", np.array([-124], dtype=np.int32))
    next(node for node in model.graph.node if node.name == "halo_conv").input.append("halo_bias_i32")

    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(output))
    return output


if __name__ == "__main__":
    print(build(TASK_ROOT / "onnx" / "task349_two_channel_candidate.onnx"))
