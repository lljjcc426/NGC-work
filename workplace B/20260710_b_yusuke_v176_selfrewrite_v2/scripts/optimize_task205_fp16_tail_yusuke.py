from __future__ import annotations

from collections import Counter

import onnx
from onnx import TensorProto, helper

import optimize_equivalent as oe


def _set_cast_dtype(node: onnx.NodeProto, dtype: int) -> bool:
    for attr in node.attribute:
        if attr.name == "to":
            if int(attr.i) == dtype:
                return False
            attr.i = dtype
            return True
    return False


def transform(model: onnx.ModelProto) -> tuple[onnx.ModelProto, Counter]:
    current = onnx.ModelProto.FromString(model.SerializeToString())
    outputs = {out for node in current.graph.node for out in node.output if out}
    required = {
        "safe_name_43",
        "safe_name_60",
        "safe_name_61",
        "safe_name_66",
        "safe_name_67",
        "safe_name_68",
        "safe_name_69",
        "safe_name_70",
        "safe_name_73",
        "safe_name_76",
        "safe_name_79",
        "output",
    }
    if not required.issubset(outputs):
        return current, Counter()

    stats = Counter()
    new_nodes: list[onnx.NodeProto] = []
    for node in current.graph.node:
        copied = onnx.NodeProto.FromString(node.SerializeToString())
        first_out = copied.output[0] if copied.output else ""

        if first_out in {"safe_name_66", "safe_name_67"} and copied.op_type == "Cast":
            if _set_cast_dtype(copied, TensorProto.FLOAT16):
                stats["tail_mask_cast_f16"] += 1

        if first_out == "safe_name_43":
            new_nodes.append(copied)
            new_nodes.append(
                helper.make_node(
                    "Cast",
                    ["safe_name_43"],
                    ["task205_s43_f16"],
                    name="task205_s43_f16",
                    to=TensorProto.FLOAT16,
                )
            )
            stats["local_cast_f16"] += 1
            continue
        if first_out == "safe_name_60":
            new_nodes.append(copied)
            new_nodes.append(
                helper.make_node(
                    "Cast",
                    ["safe_name_60"],
                    ["task205_s60_f16"],
                    name="task205_s60_f16",
                    to=TensorProto.FLOAT16,
                )
            )
            stats["local_cast_f16"] += 1
            continue
        if first_out == "safe_name_61":
            new_nodes.append(copied)
            new_nodes.append(
                helper.make_node(
                    "Cast",
                    ["safe_name_61"],
                    ["task205_s61_f16"],
                    name="task205_s61_f16",
                    to=TensorProto.FLOAT16,
                )
            )
            stats["local_cast_f16"] += 1
            continue
        if first_out == "safe_name_70":
            new_nodes.append(copied)
            new_nodes.append(
                helper.make_node(
                    "Cast",
                    ["safe_name_70"],
                    ["task205_s70_f16"],
                    name="task205_s70_f16",
                    to=TensorProto.FLOAT16,
                )
            )
            stats["local_cast_f16"] += 1
            continue

        if first_out == "safe_name_68" and copied.op_type == "Mul":
            copied.input[1] = "task205_s60_f16"
            stats["tail_mul_f16"] += 1
        elif first_out == "safe_name_69" and copied.op_type == "Mul":
            copied.input[1] = "task205_s61_f16"
            stats["tail_mul_f16"] += 1
        elif first_out == "safe_name_71" and copied.op_type == "Unsqueeze":
            copied.input[0] = "task205_s43_f16"
            stats["tail_unsqueeze_f16"] += 1
        elif first_out == "safe_name_72" and copied.op_type == "Unsqueeze":
            copied.input[0] = "task205_s70_f16"
            stats["tail_unsqueeze_f16"] += 1

        new_nodes.append(copied)

    if not stats:
        return current, Counter()

    del current.graph.node[:]
    current.graph.node.extend(new_nodes)
    for out in current.graph.output:
        if out.name == "output":
            out.type.tensor_type.elem_type = TensorProto.FLOAT16
    del current.graph.value_info[:]
    stats["dead_nodes"] += oe.prune_dead(current)
    stats["initializers"] += oe.prune_initializers(current)
    onnx.checker.check_model(current, full_check=True)
    return current, stats
