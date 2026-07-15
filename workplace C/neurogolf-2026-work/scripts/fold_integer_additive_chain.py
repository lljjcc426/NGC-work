from __future__ import annotations

from collections import Counter

import numpy as np
import onnx
from onnx import numpy_helper


INTEGER_TYPES = {
    onnx.TensorProto.INT8,
    onnx.TensorProto.UINT8,
    onnx.TensorProto.INT16,
    onnx.TensorProto.UINT16,
    onnx.TensorProto.INT32,
    onnx.TensorProto.UINT32,
    onnx.TensorProto.INT64,
    onnx.TensorProto.UINT64,
}


def _types(model: onnx.ModelProto) -> dict[str, int]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    result = {
        value.name: value.type.tensor_type.elem_type
        for value in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
        if value.type.HasField("tensor_type")
    }
    result.update({value.name: value.data_type for value in inferred.graph.initializer})
    return result


def _parts(node: onnx.NodeProto, constants: dict[str, np.ndarray]):
    if node.op_type not in {"Add", "Sub"} or len(node.input) != 2:
        return None
    constant_positions = [index for index, name in enumerate(node.input) if name in constants]
    if len(constant_positions) != 1:
        return None
    constant_position = constant_positions[0]
    dynamic_position = 1 - constant_position
    constant = constants[node.input[constant_position]]
    if node.op_type == "Add":
        return node.input[dynamic_position], 1, constant
    if dynamic_position == 0:
        return node.input[0], 1, np.negative(constant, dtype=constant.dtype)
    return node.input[1], -1, constant


def _combine(first_sign: int, first_constant: np.ndarray, second_sign: int, second_constant: np.ndarray):
    if second_sign == 1:
        constant = np.add(first_constant, second_constant, dtype=first_constant.dtype)
    else:
        constant = np.subtract(second_constant, first_constant, dtype=first_constant.dtype)
    return first_sign * second_sign, constant


def _replace_all(model: onnx.ModelProto, old: str, new: str) -> None:
    for node in model.graph.node:
        for index, value in enumerate(node.input):
            if value == old:
                node.input[index] = new
    for output in model.graph.output:
        if output.name == old:
            output.name = new


def fold(model: onnx.ModelProto) -> int:
    """Compose one-use integer Add/Sub chains using exact modular arithmetic."""
    changed = 0
    serial = 0
    while True:
        types = _types(model)
        constants = {value.name: numpy_helper.to_array(value) for value in model.graph.initializer}
        consumers = Counter(value for node in model.graph.node for value in node.input if value)
        producers = {value: node for node in model.graph.node for value in node.output if value}
        match = None
        for outer in model.graph.node:
            outer_parts = _parts(outer, constants)
            if outer_parts is None:
                continue
            inner_output, second_sign, second_constant = outer_parts
            inner = producers.get(inner_output)
            if inner is None or consumers[inner_output] != 1:
                continue
            inner_parts = _parts(inner, constants)
            if inner_parts is None:
                continue
            source, first_sign, first_constant = inner_parts
            source_type = types.get(source)
            if (
                source_type not in INTEGER_TYPES
                or first_constant.dtype != second_constant.dtype
                or onnx.helper.np_dtype_to_tensor_dtype(first_constant.dtype) != source_type
            ):
                continue
            try:
                sign, constant = _combine(first_sign, first_constant, second_sign, second_constant)
            except ValueError:
                continue
            match = outer, inner, source, sign, np.asarray(constant)
            break
        if match is None:
            break

        outer, inner, source, sign, constant = match
        if sign == 1 and not np.any(constant):
            _replace_all(model, outer.output[0], source)
            model.graph.node.remove(inner)
            model.graph.node.remove(outer)
        else:
            serial += 1
            constant_name = f"{outer.output[0]}__folded_constant_{serial}"
            model.graph.initializer.append(numpy_helper.from_array(constant, name=constant_name))
            inputs = [source, constant_name] if sign == 1 else [constant_name, source]
            replacement = onnx.helper.make_node(
                "Add" if sign == 1 else "Sub",
                inputs,
                list(outer.output),
                name=outer.name,
            )
            nodes = []
            for node in model.graph.node:
                if node is inner:
                    continue
                nodes.append(replacement if node is outer else node)
            del model.graph.node[:]
            model.graph.node.extend(nodes)

        live = {value for node in model.graph.node for value in node.input if value}
        kept = [value for value in model.graph.initializer if value.name in live]
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept)
        changed += 1
    return changed
