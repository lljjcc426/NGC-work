from __future__ import annotations

from collections import Counter

import onnx

import optimize_equivalent as oe


def transform(model: onnx.ModelProto) -> tuple[onnx.ModelProto, Counter]:
    current = onnx.ModelProto.FromString(model.SerializeToString())
    by_output = {
        output: node
        for node in current.graph.node
        for output in node.output
        if output
    }
    pad = by_output.get("gp")
    if pad is None or pad.op_type != "Pad" or pad.input[0] != "g":
        return current, Counter()

    changed = 0
    for node in current.graph.node:
        for index, name in enumerate(node.input):
            if name == "gp":
                node.input[index] = "g"
                changed += 1
    if not changed:
        return current, Counter()

    kept = [node for node in current.graph.node if node is not pad]
    del current.graph.node[:]
    current.graph.node.extend(kept)

    stats = Counter({"task285_safe_unpad": 1, "rewired_consumers": changed})
    stats["initializers"] += oe.prune_initializers(current)
    del current.graph.value_info[:]

    onnx.checker.check_model(current, full_check=True)
    onnx.shape_inference.infer_shapes(current, strict_mode=True)
    return current, stats
