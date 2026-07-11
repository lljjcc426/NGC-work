from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import onnx
from onnx import helper, numpy_helper


def build(source: Path, output: Path) -> Path:
    model = deepcopy(onnx.load(source))
    conv = model.graph.node[0]
    if conv.op_type != "Conv" or conv.input[1] != "cw":
        raise RuntimeError("unexpected task383 source graph")

    weight = next(item for item in model.graph.initializer if item.name == "cw")
    compact = numpy_helper.to_array(weight)[:, :, :1, :1]
    weight.CopyFrom(numpy_helper.from_array(compact, name="cw"))

    del conv.attribute[:]
    conv.attribute.extend(
        [
            helper.make_attribute("kernel_shape", [1, 1]),
            helper.make_attribute("pads", [0, 0, -6, -6]),
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.source, args.output))


if __name__ == "__main__":
    main()
