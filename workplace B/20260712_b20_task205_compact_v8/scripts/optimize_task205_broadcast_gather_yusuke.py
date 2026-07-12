from __future__ import annotations

from collections import Counter, defaultdict

import onnx
from onnx import TensorProto, helper


def transform(model: onnx.ModelProto) -> tuple[onnx.ModelProto, Counter]:
    current = onnx.ModelProto.FromString(model.SerializeToString())
    by_output = {
        output: (index, node)
        for index, node in enumerate(current.graph.node)
        for output in node.output
        if output
    }
    users: dict[str, list[tuple[int, onnx.NodeProto]]] = defaultdict(list)
    for index, node in enumerate(current.graph.node):
        for name in node.input:
            if name:
                users[name].append((index, node))

    chains = [
        {
            "source": "safe_name_50",
            "gathered": "safe_name_58",
            "squeezed": "safe_name_60",
            "cast": "task205_s60_f16",
            "mask": "safe_name_66",
            "mul": "safe_name_68",
            "unsqueezed": "safe_name_75",
        },
        {
            "source": "safe_name_51",
            "gathered": "safe_name_59",
            "squeezed": "safe_name_61",
            "cast": "task205_s61_f16",
            "mask": "safe_name_67",
            "mul": "safe_name_69",
            "unsqueezed": "safe_name_78",
        },
    ]
    replacements: dict[int, list[onnx.NodeProto]] = {}
    for chain in chains:
        gather_index, gather = by_output[chain["gathered"]]
        squeeze_index, squeeze = by_output[chain["squeezed"]]
        cast_index, cast = by_output[chain["cast"]]
        mul_index, mul = by_output[chain["mul"]]
        unsqueeze_index, unsqueeze = by_output[chain["unsqueezed"]]
        if (
            gather.op_type != "Gather"
            or gather.input[0] != chain["source"]
            or squeeze.op_type != "Squeeze"
            or cast.op_type != "Cast"
            or mul.op_type != "Mul"
            or unsqueeze.op_type != "Unsqueeze"
            or len(users[chain["gathered"]]) != 1
            or len(users[chain["squeezed"]]) != 1
            or len(users[chain["mul"]]) != 1
        ):
            return current, Counter()

        axis = next(
            (onnx.helper.get_attribute_value(attr) for attr in gather.attribute if attr.name == "axis"),
            0,
        )
        cast_source = f"{chain['cast']}_source_f16"
        selected_3d = f"{chain['cast']}_selected_3d"
        replacements[gather_index] = [
            helper.make_node(
                "Cast",
                [chain["source"]],
                [cast_source],
                to=TensorProto.FLOAT16,
                name=cast_source,
            ),
            helper.make_node(
                "Gather",
                [cast_source, gather.input[1]],
                [selected_3d],
                axis=int(axis),
                name=selected_3d,
            ),
        ]
        replacements[squeeze_index] = []
        replacements[cast_index] = []
        replacements[mul_index] = [
            helper.make_node(
                "Mul",
                [chain["mask"], selected_3d],
                [chain["unsqueezed"]],
                name=chain["unsqueezed"],
            )
        ]
        replacements[unsqueeze_index] = []

    nodes: list[onnx.NodeProto] = []
    for index, node in enumerate(current.graph.node):
        if index in replacements:
            nodes.extend(replacements[index])
        else:
            nodes.append(onnx.NodeProto.FromString(node.SerializeToString()))
    del current.graph.node[:]
    current.graph.node.extend(nodes)
    del current.graph.value_info[:]

    onnx.checker.check_model(current, full_check=True)
    onnx.shape_inference.infer_shapes(current, strict_mode=True)
    return current, Counter({"task205_broadcast_gather": 2})
