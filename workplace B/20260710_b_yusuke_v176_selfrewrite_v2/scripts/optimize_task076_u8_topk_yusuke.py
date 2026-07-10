from __future__ import annotations

from collections import Counter

import onnx
from onnx import TensorProto

import optimize_equivalent as oe


TARGET_CAST_OUTPUTS = {
    "target_red_flat",
    "source_hidden_flat",
    "source_visible_flat",
}


def transform(model: onnx.ModelProto) -> tuple[onnx.ModelProto, Counter]:
    current = onnx.ModelProto.FromString(model.SerializeToString())
    outputs = {out for node in current.graph.node for out in node.output if out}
    required = TARGET_CAST_OUTPUTS | {
        "target_topk_values",
        "source_hidden_topk_values",
        "source_visible_topk_values",
    }
    if not required.issubset(outputs):
        return current, Counter()

    stats = Counter()
    for node in current.graph.node:
        if node.op_type != "Cast" or not node.output or node.output[0] not in TARGET_CAST_OUTPUTS:
            continue
        for attr in node.attribute:
            if attr.name == "to" and int(attr.i) != TensorProto.UINT8:
                attr.i = TensorProto.UINT8
                stats["mask_cast_u8"] += 1

    if not stats:
        return current, Counter()

    del current.graph.value_info[:]
    stats["dead_nodes"] += oe.prune_dead(current)
    stats["initializers"] += oe.prune_initializers(current)
    onnx.checker.check_model(current, full_check=True)
    return current, stats
