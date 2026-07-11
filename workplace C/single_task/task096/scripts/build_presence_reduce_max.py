from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import onnx
from onnx import helper, numpy_helper


def build(source: Path, output: Path) -> Path:
    model = deepcopy(onnx.load(source))
    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    changed = 0
    for node in model.graph.node:
        if node.output and node.output[0] in {"row_sum_full", "col_sum_full"}:
            if node.op_type != "ReduceSum":
                raise RuntimeError(f"unexpected producer for {node.output[0]}")
            axes_name = node.input[1]
            axes = [int(value) for value in initializers[axes_name].reshape(-1)]
            node.op_type = "ReduceMax"
            del node.input[1:]
            node.attribute.extend([helper.make_attribute("axes", axes)])
            changed += 1
    if changed != 2:
        raise RuntimeError(f"expected two presence reductions, changed {changed}")
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
