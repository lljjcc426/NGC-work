from __future__ import annotations

import argparse
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


def build(output: Path, collision_budget: int) -> Path:
    """Merge width-2/4 channels using a collision-safe amplitude code."""
    if collision_budget != 124:
        raise ValueError("the proven integer code uses collision_budget=124")
    model = onnx.load(str(SOURCE))
    h_node = next(node for node in model.graph.node if node.name == "h_conv")
    halo_node = next(node for node in model.graph.node if node.name == "halo_conv")
    old_h = next(
        numpy_helper.to_array(value)
        for value in model.graph.initializer
        if value.name == "h_kernel_combined_i8"
    )
    old_h_bias = next(
        numpy_helper.to_array(value)
        for value in model.graph.initializer
        if value.name == "h_bias_combined_i32"
    )
    old_halo = next(
        numpy_helper.to_array(value)
        for value in model.graph.initializer
        if value.name == "halo_weight_i8"
    )

    # Leave a margin of three because the downstream color logic expects halo
    # values to clamp to color 3, not merely become positive.
    amplitude = collision_budget + 3
    merged_h = np.zeros((4, 1, 2, 11), dtype=np.int8)
    # This integer filter is the exact collision-aware union of the width-2
    # and width-4 detectors. It emits 1 and `amplitude`, respectively, while
    # all shifted/longer public run patterns remain non-positive.
    merged_h[0, 0, 1] = np.array(
        [-127, 0, 127, -1, 127, -127, 0, 0, 0, 0, 0],
        dtype=np.int8,
    )
    merged_h[1:] = old_h[2:]
    merged_bias = np.concatenate((np.array([-126], dtype=np.int32), old_h_bias[2:]))

    small = old_halo[0, 0] > 0
    large = old_halo[0, 1] > 0
    if np.any(small & ~large):
        raise RuntimeError("width-2 support is not nested inside width-4 support")
    merged_halo = np.zeros((1, 4, 11, 20), dtype=np.int8)
    merged_halo[0, 0, large & ~small] = 1
    merged_halo[0, 0, small] = amplitude
    for output_channel, source_channel in enumerate(range(2, 5), start=1):
        merged_halo[0, output_channel, old_halo[0, source_channel] > 0] = amplitude

    replace_initializer(model, "h_kernel_combined_i8", merged_h)
    replace_initializer(model, "h_bias_combined_i32", merged_bias)
    replace_initializer(model, "halo_weight_i8", merged_halo)
    replace_initializer(model, "halo_bias_i32", np.array([-collision_budget], dtype=np.int32))
    halo_node.input.append("halo_bias_i32")

    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(output))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-budget", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.output, args.collision_budget))


if __name__ == "__main__":
    main()
