from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


TASK_ROOT = Path(__file__).resolve().parents[1]
SOURCE = TASK_ROOT / "onnx" / "task349_width29_toptrim_base.onnx"


def build_spatial_base() -> Path:
    script = Path(__file__).resolve().parent / "build_width29_halo.py"
    spec = importlib.util.spec_from_file_location("task349_spatial_base", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build(SOURCE, trim_top=True)


def replace_initializer(model: onnx.ModelProto, name: str, value: np.ndarray) -> None:
    tensor = numpy_helper.from_array(value, name=name)
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name == name:
            model.graph.initializer[index].CopyFrom(tensor)
            return
    model.graph.initializer.append(tensor)


def build(output: Path) -> Path:
    """Merge widths 2/4 and 6/8 under one 124-collision halo threshold."""
    model = onnx.load(str(build_spatial_base()))
    old_h = next(
        numpy_helper.to_array(value)
        for value in model.graph.initializer
        if value.name == "h_kernel_combined_i8"
    )
    old_halo = next(
        numpy_helper.to_array(value)
        for value in model.graph.initializer
        if value.name == "halo_weight_i8"
    )

    detector = np.zeros((3, 1, 2, 11), dtype=np.int8)
    detector[0, 0, 1] = [-127, 0, 127, -1, 127, -127, 0, 0, 0, 0, 0]
    detector[1, 0, 1] = [-127, 127, 0, 127, -127, 127, 0, 126, 0, -127, -127]
    detector[2] = old_h[4]
    detector_bias = np.array([-126, -253, -9], dtype=np.int32)

    halo = np.zeros((1, 3, 11, 20), dtype=np.int8)
    for target_channel, small_index, large_index in ((0, 0, 1), (1, 2, 3)):
        small = old_halo[0, small_index] > 0
        large = old_halo[0, large_index] > 0
        halo[0, target_channel, large & ~small] = 1
        halo[0, target_channel, small] = 127
    halo[0, 2, old_halo[0, 4] > 0] = 127

    replace_initializer(model, "h_kernel_combined_i8", detector)
    replace_initializer(model, "h_bias_combined_i32", detector_bias)
    replace_initializer(model, "halo_weight_i8", halo)
    replace_initializer(model, "halo_bias_i32", np.array([-124], dtype=np.int32))
    halo_node = next(node for node in model.graph.node if node.name == "halo_conv")
    halo_node.input.append("halo_bias_i32")

    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(output))
    return output


if __name__ == "__main__":
    print(build(TASK_ROOT / "onnx" / "task349_candidate.onnx"))
