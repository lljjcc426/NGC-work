from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def _equation(node: onnx.NodeProto) -> onnx.AttributeProto:
    return next(attr for attr in node.attribute if attr.name == "equation")


def build(source: Path, output: Path, left_name: str, right_name: str) -> Path:
    model = onnx.load(source)
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    if left_name not in arrays or right_name not in arrays:
        raise RuntimeError("requested initializer pair is missing")

    uses = {
        name: [(node, index) for node in model.graph.node for index, value in enumerate(node.input) if value == name]
        for name in (left_name, right_name)
    }
    if len(uses[left_name]) != 1 or len(uses[right_name]) != 1:
        raise RuntimeError(f"pair must be single-use: { {name: len(value) for name, value in uses.items()} }")
    node, left_index = uses[left_name][0]
    right_node, right_index = uses[right_name][0]
    if node is not right_node or node.op_type != "Einsum":
        raise RuntimeError("pair must be consumed by the same Einsum node")

    attribute = _equation(node)
    lhs, rhs = attribute.s.decode("ascii").split("->", 1)
    terms = lhs.split(",")
    left_term = terms[left_index]
    right_term = terms[right_index]
    if "." in left_term or "." in right_term:
        raise RuntimeError("ellipsis equations are not supported")
    counts = Counter("".join(terms))
    shared = [label for label in left_term if label in right_term and label not in rhs and counts[label] == 2]
    if len(shared) != 1:
        raise RuntimeError(f"expected one private contracted label, found {shared}")
    label = shared[0]
    left_axis = left_term.index(label)
    right_axis = right_term.index(label)
    if any(value == label for index, term in enumerate(terms) if index not in {left_index, right_index} for value in term):
        raise RuntimeError("contracted label is shared with another operand")

    left = arrays[left_name]
    right = arrays[right_name]
    contracted = np.tensordot(left, right, axes=([left_axis], [right_axis]))
    contracted_term = left_term[:left_axis] + left_term[left_axis + 1 :] + right_term[:right_axis] + right_term[right_axis + 1 :]
    if len(contracted_term) != contracted.ndim or len(set(contracted_term)) != len(contracted_term):
        raise RuntimeError(f"unsupported contracted term: {contracted_term}")

    new_name = f"{left_name}_{right_name}_contracted"
    first, second = sorted((left_index, right_index))
    inputs = list(node.input)
    new_terms = list(terms)
    inputs[first] = new_name
    new_terms[first] = contracted_term
    del inputs[second]
    del new_terms[second]
    del node.input[:]
    node.input.extend(inputs)
    attribute.s = (",".join(new_terms) + "->" + rhs).encode("ascii")

    kept = [item for item in model.graph.initializer if item.name not in {left_name, right_name}]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.initializer.append(numpy_helper.from_array(contracted, new_name))
    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(inferred, output)
    reloaded = onnx.load(output)
    onnx.checker.check_model(reloaded, full_check=True)
    onnx.shape_inference.infer_shapes(reloaded, strict_mode=True, data_prop=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Precontract a private single-use initializer pair inside a terminal Einsum.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    args = parser.parse_args()
    print(build(args.source, args.output, args.left, args.right))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
