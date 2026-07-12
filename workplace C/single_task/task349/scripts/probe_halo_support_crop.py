from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import numpy_helper


TASK_DIR = Path(__file__).resolve().parents[1]
SOURCE = TASK_DIR / "onnx" / "task349_candidate.onnx"


def replace_initializer(model: onnx.ModelProto, name: str, value) -> None:
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name == name:
            model.graph.initializer[index].CopyFrom(numpy_helper.from_array(value, name=name))
            return
    raise KeyError(name)


def build(output: Path, top: int, bottom: int, left: int, right: int) -> Path:
    model = onnx.load(str(SOURCE))
    node = next(node for node in model.graph.node if node.name == "halo_conv")
    weight = next(
        numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
        if initializer.name == "halo_weight_i8"
    )
    height, width = weight.shape[-2:]
    cropped = weight[:, :, top : height - bottom if bottom else height, left : width - right if right else width].copy()
    replace_initializer(model, "halo_weight_i8", cropped)
    attrs = {attribute.name: attribute for attribute in node.attribute}
    attrs["kernel_shape"].ints[:] = list(cropped.shape[-2:])
    pads = list(attrs["pads"].ints)
    pads[:] = [pads[0] - top, pads[1] - left, pads[2] - bottom, pads[3] - right]
    if min(pads) < 0:
        raise ValueError(f"negative pads after crop: {pads}")
    attrs["pads"].ints[:] = pads
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(output))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=0)
    parser.add_argument("--bottom", type=int, default=0)
    parser.add_argument("--left", type=int, default=0)
    parser.add_argument("--right", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.output, args.top, args.bottom, args.left, args.right))


if __name__ == "__main__":
    main()
