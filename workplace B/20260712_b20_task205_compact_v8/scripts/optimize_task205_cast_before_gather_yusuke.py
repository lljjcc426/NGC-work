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
        ("safe_name_50", "safe_name_58", "safe_name_60", "task205_s60_f16"),
        ("safe_name_51", "safe_name_59", "safe_name_61", "task205_s61_f16"),
    ]
    replacements: dict[int, list[onnx.NodeProto]] = {}
    for source, gathered, squeezed, final in chains:
        gather_item = by_output.get(gathered)
        squeeze_item = by_output.get(squeezed)
        cast_item = by_output.get(final)
        if gather_item is None or squeeze_item is None or cast_item is None:
            return current, Counter()
        gather_index, gather = gather_item
        squeeze_index, squeeze = squeeze_item
        cast_index, cast = cast_item
        if (
            gather.op_type != "Gather"
            or gather.input[0] != source
            or squeeze.op_type != "Squeeze"
            or list(squeeze.input) != [gathered]
            or cast.op_type != "Cast"
            or list(cast.input) != [squeezed]
            or len(users[gathered]) != 1
            or len(users[squeezed]) != 1
        ):
            return current, Counter()

        axis = next(
            (onnx.helper.get_attribute_value(attr) for attr in gather.attribute if attr.name == "axis"),
            0,
        )
        axes = next(
            (onnx.helper.get_attribute_value(attr) for attr in squeeze.attribute if attr.name == "axes"),
            None,
        )
        if axes is None:
            return current, Counter()

        cast_source = f"{final}_source_f16"
        gather_f16 = f"{final}_gather_f16"
        replacements[gather_index] = [
            helper.make_node(
                "Cast",
                [source],
                [cast_source],
                to=TensorProto.FLOAT16,
                name=cast_source,
            ),
            helper.make_node(
                "Gather",
                [cast_source, gather.input[1]],
                [gather_f16],
                axis=int(axis),
                name=gather_f16,
            ),
            helper.make_node(
                "Squeeze",
                [gather_f16],
                [final],
                axes=list(axes),
                name=final,
            ),
        ]
        replacements[squeeze_index] = []
        replacements[cast_index] = []

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
    return current, Counter({"task205_cast_before_gather": 2})
