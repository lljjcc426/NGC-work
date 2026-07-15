from __future__ import annotations

from collections import Counter

import onnx


def _replace_input(model: onnx.ModelProto, old: str, new: str) -> None:
    for node in model.graph.node:
        for index, value in enumerate(node.input):
            if value == old:
                node.input[index] = new
    for output in model.graph.output:
        if output.name == old:
            output.name = new


def fold(model: onnx.ModelProto) -> int:
    """Fold nested Where nodes when they share the selected data branch.

    Where(a, x, Where(b, x, y)) == Where(Or(a, b), x, y)
    Where(a, Where(b, x, y), y) == Where(And(a, b), x, y)
    """
    changed = 0
    while True:
        consumers = Counter(value for node in model.graph.node for value in node.input if value)
        producers = {value: node for node in model.graph.node for value in node.output if value}
        replacement: tuple[onnx.NodeProto, onnx.NodeProto, onnx.NodeProto] | None = None

        for outer in model.graph.node:
            if outer.op_type != "Where" or len(outer.input) != 3 or len(outer.output) != 1:
                continue

            false_inner = producers.get(outer.input[2])
            if (
                false_inner is not None
                and false_inner.op_type == "Where"
                and len(false_inner.input) == 3
                and len(false_inner.output) == 1
                and consumers[false_inner.output[0]] == 1
                and outer.input[1] == false_inner.input[1]
            ):
                condition = outer.output[0] + "__or_condition"
                logic = onnx.helper.make_node(
                    "Or", [outer.input[0], false_inner.input[0]], [condition],
                    name=(outer.name or "where") + "__or_condition",
                )
                folded = onnx.helper.make_node(
                    "Where", [condition, outer.input[1], false_inner.input[2]],
                    list(outer.output), name=outer.name,
                )
                replacement = (outer, false_inner, logic, folded)
                break

            true_inner = producers.get(outer.input[1])
            if (
                true_inner is not None
                and true_inner.op_type == "Where"
                and len(true_inner.input) == 3
                and len(true_inner.output) == 1
                and consumers[true_inner.output[0]] == 1
                and outer.input[2] == true_inner.input[2]
            ):
                condition = outer.output[0] + "__and_condition"
                logic = onnx.helper.make_node(
                    "And", [outer.input[0], true_inner.input[0]], [condition],
                    name=(outer.name or "where") + "__and_condition",
                )
                folded = onnx.helper.make_node(
                    "Where", [condition, true_inner.input[1], outer.input[2]],
                    list(outer.output), name=outer.name,
                )
                replacement = (outer, true_inner, logic, folded)
                break

        if replacement is None:
            break

        outer, inner, logic, folded = replacement
        nodes = []
        for node in model.graph.node:
            if node is inner:
                continue
            if node is outer:
                nodes.extend((logic, folded))
                continue
            nodes.append(node)
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        changed += 1
    return changed
