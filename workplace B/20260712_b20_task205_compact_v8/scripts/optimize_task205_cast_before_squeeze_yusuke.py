from __future__ import annotations

from collections import Counter

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
    pairs = [
        ("safe_name_58", "safe_name_60", "task205_s60_f16"),
        ("safe_name_59", "safe_name_61", "task205_s61_f16"),
    ]
    replacements: dict[int, list[onnx.NodeProto]] = {}
    for source, squeezed, final in pairs:
        squeeze_item = by_output.get(squeezed)
        cast_item = by_output.get(final)
        if squeeze_item is None or cast_item is None:
            return current, Counter()
        squeeze_index, squeeze = squeeze_item
        cast_index, cast = cast_item
        if (
            squeeze.op_type != "Squeeze"
            or list(squeeze.input) != [source]
            or cast.op_type != "Cast"
            or list(cast.input) != [squeezed]
        ):
            return current, Counter()
        axes = next(
            (onnx.helper.get_attribute_value(attr) for attr in squeeze.attribute if attr.name == "axes"),
            None,
        )
        if axes is None:
            return current, Counter()
        temporary = f"{final}_pre_squeeze"
        replacements[squeeze_index] = [
            helper.make_node(
                "Cast",
                [source],
                [temporary],
                to=TensorProto.FLOAT16,
                name=temporary,
            ),
            helper.make_node(
                "Squeeze",
                [temporary],
                [final],
                axes=list(axes),
                name=final,
            ),
        ]
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
    return current, Counter({"task205_cast_before_squeeze": 2})
