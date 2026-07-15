from __future__ import annotations

from collections import Counter

import onnx


OPS = {"And", "Or", "Min", "Max"}


def fold(model: onnx.ModelProto) -> int:
    """Collapse op(x, op(x, y)) to op(x, y) for idempotent associative ops."""
    changed = 0
    while True:
        consumers = Counter(value for node in model.graph.node for value in node.input if value)
        producers = {value: node for node in model.graph.node for value in node.output if value}
        match: tuple[onnx.NodeProto, onnx.NodeProto, str, str] | None = None
        for outer in model.graph.node:
            if outer.op_type not in OPS or len(outer.input) != 2 or len(outer.output) != 1:
                continue
            for inner_index, shared_index in ((0, 1), (1, 0)):
                inner = producers.get(outer.input[inner_index])
                if (
                    inner is None
                    or inner.op_type != outer.op_type
                    or len(inner.input) != 2
                    or len(inner.output) != 1
                    or consumers[inner.output[0]] != 1
                ):
                    continue
                shared = outer.input[shared_index]
                if shared == inner.input[0]:
                    match = (outer, inner, shared, inner.input[1])
                    break
                if shared == inner.input[1]:
                    match = (outer, inner, shared, inner.input[0])
                    break
            if match is not None:
                break
        if match is None:
            break
        outer, inner, left, right = match
        outer.input[:] = [left, right]
        model.graph.node.remove(inner)
        changed += 1
    return changed
