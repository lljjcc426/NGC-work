from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def build(parent_path: Path, output_path: Path) -> Path:
    model = onnx.load(parent_path)
    nodes = list(model.graph.node)
    if [nodes[index].op_type for index in (20, 21, 22, 23, 24, 25)] != [
        "Min",
        "Min",
        "Max",
        "Cast",
        "Pad",
        "Where",
    ]:
        raise RuntimeError("unexpected task064 terminal graph")
    if list(nodes[22].input) != ["hline", "vline"]:
        raise RuntimeError("unexpected task064 line inputs")

    equal = helper.make_node("Equal", ["hline", "vline"], ["not_line"])
    pad = helper.make_node(
        "Pad",
        ["not_line", "pads_hw", "pad_true", "ax23"],
        ["not_line30"],
        mode="constant",
    )
    where = helper.make_node(
        "Where",
        ["not_line30", "input", "marker_ohf"],
        ["output"],
    )
    del model.graph.node[22:]
    model.graph.node.extend([equal, pad, where])
    model.graph.initializer.append(
        numpy_helper.from_array(np.asarray(True, dtype=np.bool_), name="pad_true")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.save(model, output_path)
    reloaded = onnx.load(output_path)
    onnx.checker.check_model(reloaded, full_check=True)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.parent, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
