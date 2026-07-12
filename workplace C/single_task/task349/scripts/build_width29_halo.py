from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


TASK_DIR = Path(__file__).resolve().parents[1]
SOURCE = Path(
    r"E:/kagglegolf/submissions/candidates/"
    r"GOLF_20260711_096_v95_plus_4_compact/onnx/task349.onnx"
)
OUTPUT = TASK_DIR / "onnx" / "task349_width29_halo.onnx"


def set_ints(node: onnx.NodeProto, name: str, values: list[int]) -> None:
    attribute = next(attribute for attribute in node.attribute if attribute.name == name)
    attribute.ints[:] = values


def replace_initializer(model: onnx.ModelProto, name: str, value: np.ndarray) -> None:
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name == name:
            model.graph.initializer[index].CopyFrom(numpy_helper.from_array(value, name=name))
            return
    raise KeyError(name)


def build(output_path: Path = OUTPUT, trim_boundary_rows: bool = False, trim_top: bool = False, trim_bottom: bool = False) -> Path:
    """Remove the structurally impossible last column of the width map.

    The horizontal detector emits at the left edge of a color-9 rectangle.
    Since every valid rectangle is at least two cells wide, column 29 can never
    be a left edge in the fixed 30-column canvas. Reducing the detector's right
    pad from 9 to 8 makes its output width 29. Increasing the following halo
    convolution's right pad from 5 to 6 restores the original 30-column output
    without shifting any receptive field.
    """
    model = onnx.load(str(SOURCE))
    horizontal = next(node for node in model.graph.node if node.name == "h_conv")
    halo = next(node for node in model.graph.node if node.name == "halo_conv")
    set_ints(horizontal, "pads", [0, 1, 0, 8])
    set_ints(halo, "pads", [5, 14, 5, 6])
    if trim_boundary_rows or trim_top or trim_bottom:
        kernel = next(
            numpy_helper.to_array(initializer)
            for initializer in model.graph.initializer
            if initializer.name == "h_kernel_combined_i8"
        )
        kernel_height = 3 if trim_boundary_rows else 2
        expanded = np.zeros((kernel.shape[0], kernel.shape[1], kernel_height, kernel.shape[3]), dtype=kernel.dtype)
        source_row = 1 if trim_boundary_rows or trim_top else 0
        expanded[:, :, source_row : source_row + 1, :] = kernel
        replace_initializer(model, "h_kernel_combined_i8", expanded)
        set_ints(horizontal, "kernel_shape", [kernel_height, 11])
        set_ints(
            halo,
            "pads",
            [6 if trim_boundary_rows or trim_top else 5, 14, 6 if trim_boundary_rows or trim_bottom else 5, 6],
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(output_path))
    return output_path


if __name__ == "__main__":
    print(build())
