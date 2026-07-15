from __future__ import annotations

from collections import Counter

import numpy as np
import onnx


INTEGER_TYPES = {
    onnx.TensorProto.BOOL,
    onnx.TensorProto.INT8,
    onnx.TensorProto.UINT8,
    onnx.TensorProto.INT16,
    onnx.TensorProto.UINT16,
}


def _types(model: onnx.ModelProto) -> dict[str, int]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    result: dict[str, int] = {}
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        if value.type.HasField("tensor_type"):
            result[value.name] = value.type.tensor_type.elem_type
    for value in inferred.graph.initializer:
        result[value.name] = value.data_type
    return result


def _cast_is_equivalent(source_type: int, middle_type: int, final_type: int) -> bool:
    if source_type not in INTEGER_TYPES:
        return False
    source_dtype = onnx.helper.tensor_dtype_to_np_dtype(source_type)
    middle_dtype = onnx.helper.tensor_dtype_to_np_dtype(middle_type)
    final_dtype = onnx.helper.tensor_dtype_to_np_dtype(final_type)
    if source_type == onnx.TensorProto.BOOL:
        values = np.array([False, True], dtype=source_dtype)
    else:
        info = np.iinfo(source_dtype)
        values = np.arange(info.min, info.max + 1, dtype=np.int64).astype(source_dtype)
    with np.errstate(all="ignore"):
        via_middle = values.astype(middle_dtype).astype(final_dtype)
        direct = values.astype(final_dtype)
    return bool(np.array_equal(via_middle, direct))


def fold(model: onnx.ModelProto) -> int:
    """Remove Cast A->B->C when exhaustive A-domain testing proves A->C equal."""
    changed = 0
    while True:
        types = _types(model)
        consumers = Counter(value for node in model.graph.node for value in node.input if value)
        producers = {value: node for node in model.graph.node for value in node.output if value}
        match: tuple[onnx.NodeProto, onnx.NodeProto] | None = None
        for second in model.graph.node:
            if second.op_type != "Cast" or len(second.input) != 1:
                continue
            first = producers.get(second.input[0])
            if (
                first is None
                or first.op_type != "Cast"
                or len(first.input) != 1
                or len(first.output) != 1
                or consumers[first.output[0]] != 1
            ):
                continue
            source_type = types.get(first.input[0])
            middle_type = types.get(first.output[0])
            final_type = types.get(second.output[0])
            if None in (source_type, middle_type, final_type):
                continue
            if _cast_is_equivalent(source_type, middle_type, final_type):
                match = first, second
                break
        if match is None:
            break
        first, second = match
        second.input[0] = first.input[0]
        model.graph.node.remove(first)
        changed += 1
    return changed
