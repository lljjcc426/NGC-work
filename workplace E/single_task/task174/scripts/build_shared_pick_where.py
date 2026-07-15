from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def build(parent_path: Path, output_path: Path) -> None:
    model = onnx.load(parent_path)
    nodes = list(model.graph.node)
    producers = {name: index for index, node in enumerate(nodes) for name in node.output}
    replacements: dict[int, onnx.NodeProto] = {}
    removed: set[int] = set()

    zero_i32 = "pick_zero_i32"
    zero_f32 = "pick_zero_f32"
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(np.asarray(0, dtype=np.int32), zero_i32),
            numpy_helper.from_array(np.asarray(0, dtype=np.float32), zero_f32),
        ]
    )
    for output, value in (("pt", "top"), ("pl", "left"), ("pH", "H_i"), ("pW", "W_i")):
        index = next(i for i, node in enumerate(nodes) if output in node.output)
        node = nodes[index]
        if node.op_type != "Mul" or "picki" not in node.input:
            raise RuntimeError(f"unexpected selector for {output}")
        replacements[index] = helper.make_node("Where", ["pick", value, zero_i32], [output], name=node.name)
    pk32_index = next(i for i, node in enumerate(nodes) if "pk32" in node.output)
    if nodes[pk32_index].op_type != "Mul":
        raise RuntimeError("unexpected float color selector")
    replacements[pk32_index] = helper.make_node("Where", ["pick", "colf", zero_f32], ["pk32"], name=nodes[pk32_index].name)
    removed.add(producers["picki"])
    removed.add(producers["pickf32"])

    rewritten = []
    for index, node in enumerate(nodes):
        if index in removed:
            continue
        rewritten.append(replacements.get(index, node))
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    used = {name for node in model.graph.node for name in node.input if name}
    kept = [initializer for initializer in model.graph.initializer if initializer.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    onnx.checker.check_model(onnx.load(output_path), full_check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Remove task174's shared numeric pick masks with exact Where selectors.")
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.parent, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
