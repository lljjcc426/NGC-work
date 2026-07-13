from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference, version_converter


HERE = Path(__file__).resolve()
PROJECT = HERE.parents[1]
WORKPLACE_C = PROJECT.parent
REPO_ROOT = WORKPLACE_C.parent
DEFAULT_ASSIGNMENTS = REPO_ROOT / "assignments" / "task_assignment_400.csv"
DEFAULT_ARCHIVE = PROJECT / "data" / "external" / "neurogolf7300_archive" / "onnx"
DEFAULT_PARENT = Path(
    r"E:/kagglegolf/submissions/candidates/GOLF_20260713_C5_05/onnx"
)
DEFAULT_ARTIFACTS = WORKPLACE_C / "artifacts" / "cd_parent_micro_sweep"
DEFAULT_OUTPUT = WORKPLACE_C / "score_docs" / "46_CD_PARENT_MICRO_SWEEP.csv"


ELEMENTWISE_OPS = {
    "Add",
    "And",
    "Div",
    "Equal",
    "Greater",
    "GreaterOrEqual",
    "Less",
    "LessOrEqual",
    "Max",
    "Min",
    "Mod",
    "Mul",
    "Or",
    "Pow",
    "Sub",
    "Where",
    "Xor",
}


def _attribute(node: onnx.NodeProto, name: str) -> onnx.AttributeProto | None:
    return next((item for item in node.attribute if item.name == name), None)


def _initializer_key(item: onnx.TensorProto) -> str:
    array = numpy_helper.to_array(item)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def deduplicate_initializers(model: onnx.ModelProto) -> int:
    canonical: dict[str, str] = {}
    replacements: dict[str, str] = {}
    kept: list[onnx.TensorProto] = []
    for item in model.graph.initializer:
        key = _initializer_key(item)
        if key in canonical:
            replacements[item.name] = canonical[key]
        else:
            canonical[key] = item.name
            kept.append(item)
    if not replacements:
        return 0
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name in replacements:
                node.input[index] = replacements[name]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return len(replacements)


def prune_initializers(model: onnx.ModelProto) -> int:
    used = {name for node in model.graph.node for name in node.input if name}
    graph_outputs = {item.name for item in model.graph.output}
    kept = [
        item
        for item in model.graph.initializer
        if item.name in used or item.name in graph_outputs
    ]
    removed = len(model.graph.initializer) - len(kept)
    if removed:
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept)
    return removed


def _constant_array(node: onnx.NodeProto) -> np.ndarray | None:
    if node.op_type != "Constant" or len(node.output) != 1:
        return None
    for attr in node.attribute:
        if attr.name == "value":
            return numpy_helper.to_array(attr.t)
        if attr.name == "value_float":
            return np.asarray(attr.f, dtype=np.float32)
        if attr.name == "value_int":
            return np.asarray(attr.i, dtype=np.int64)
        if attr.name == "value_floats":
            return np.asarray(attr.floats, dtype=np.float32)
        if attr.name == "value_ints":
            return np.asarray(attr.ints, dtype=np.int64)
    return None


def _tensor_sources(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    result = {
        item.name: numpy_helper.to_array(item)
        for item in model.graph.initializer
    }
    for node in model.graph.node:
        value = _constant_array(node)
        if value is not None:
            result[node.output[0]] = value
    return result


def _replace_source_array(model: onnx.ModelProto, name: str, value: np.ndarray) -> bool:
    for item in model.graph.initializer:
        if item.name == name:
            item.CopyFrom(numpy_helper.from_array(value, name=name))
            return True
    for node in model.graph.node:
        if node.op_type != "Constant" or not node.output or node.output[0] != name:
            continue
        del node.attribute[:]
        attr = node.attribute.add()
        attr.name = "value"
        attr.type = onnx.AttributeProto.TENSOR
        attr.t.CopyFrom(numpy_helper.from_array(value))
        return True
    return False


def deduplicate_constant_tensors(model: onnx.ModelProto) -> int:
    graph_outputs = {item.name for item in model.graph.output}
    used = {name for node in model.graph.node for name in node.input if name}
    canonical: dict[str, str] = {
        _initializer_key(item): item.name for item in model.graph.initializer
    }
    replacements: dict[str, str] = {}
    kept: list[onnx.NodeProto] = []
    removed = 0
    for node in model.graph.node:
        value = _constant_array(node)
        if value is None or node.output[0] in graph_outputs:
            kept.append(node)
            continue
        if node.output[0] not in used:
            removed += 1
            continue
        tensor = numpy_helper.from_array(value, name=node.output[0])
        key = _initializer_key(tensor)
        if key in canonical:
            replacements[node.output[0]] = canonical[key]
            removed += 1
        else:
            canonical[key] = node.output[0]
            kept.append(node)
    if replacements:
        for node in kept:
            for index, name in enumerate(node.input):
                if name in replacements:
                    node.input[index] = replacements[name]
    if removed:
        del model.graph.node[:]
        model.graph.node.extend(kept)
    return removed


def _shape_map(model: onnx.ModelProto) -> dict[str, tuple[int, ...]]:
    try:
        inferred = shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    except Exception:
        inferred = model
    result: dict[str, tuple[int, ...]] = {}
    for item in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]:
        tensor = item.type.tensor_type
        if not tensor.HasField("shape"):
            continue
        dims: list[int] = []
        valid = True
        for dim in tensor.shape.dim:
            if not dim.HasField("dim_value") or dim.dim_value <= 0:
                valid = False
                break
            dims.append(int(dim.dim_value))
        if valid:
            result[item.name] = tuple(dims)
    for item in inferred.graph.initializer:
        result[item.name] = tuple(int(value) for value in item.dims)
    return result


def _type_map(model: onnx.ModelProto) -> dict[str, int]:
    try:
        inferred = shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    except Exception:
        inferred = model
    result: dict[str, int] = {}
    for item in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]:
        tensor = item.type.tensor_type
        if tensor.elem_type:
            result[item.name] = int(tensor.elem_type)
    for item in inferred.graph.initializer:
        result[item.name] = int(item.data_type)
    return result


def _broadcast_shape(shapes: list[tuple[int, ...]]) -> tuple[int, ...] | None:
    if not shapes:
        return None
    rank = max(len(shape) for shape in shapes)
    output: list[int] = []
    for index in range(rank):
        values = []
        for shape in shapes:
            offset = rank - len(shape)
            values.append(1 if index < offset else shape[index - offset])
        non_one = {value for value in values if value != 1}
        if len(non_one) > 1:
            return None
        output.append(next(iter(non_one), 1))
    return tuple(output)


def collapse_broadcast_initializers(model: onnx.ModelProto) -> int:
    shapes = _shape_map(model)
    graph_inputs = {item.name for item in model.graph.input}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for name in node.input:
            if name:
                consumers.setdefault(name, []).append(node)

    collapsed = 0
    for tensor in model.graph.initializer:
        if tensor.name in graph_inputs:
            continue
        nodes = consumers.get(tensor.name, [])
        if not nodes or any(node.op_type not in ELEMENTWISE_OPS for node in nodes):
            continue
        array = numpy_helper.to_array(tensor)
        if array.ndim == 0 or array.size <= 1:
            continue
        candidate = array
        for axis in range(array.ndim):
            if candidate.shape[axis] <= 1:
                continue
            first = np.take(candidate, [0], axis=axis)
            if not np.array_equal(candidate, np.repeat(first, candidate.shape[axis], axis=axis)):
                continue
            trial = first
            trial_shape = tuple(int(value) for value in trial.shape)
            safe = True
            for node in nodes:
                input_shapes: list[tuple[int, ...]] = []
                for name in node.input:
                    if not name:
                        continue
                    if name == tensor.name:
                        input_shapes.append(trial_shape)
                    elif name in shapes:
                        input_shapes.append(shapes[name])
                    else:
                        safe = False
                        break
                if not safe or not node.output or node.output[0] not in shapes:
                    safe = False
                    break
                if _broadcast_shape(input_shapes) != shapes[node.output[0]]:
                    safe = False
                    break
            if safe:
                candidate = trial
        if candidate.shape != array.shape:
            tensor.CopyFrom(numpy_helper.from_array(candidate, name=tensor.name))
            collapsed += int(array.size - candidate.size)
            shapes[tensor.name] = tuple(int(value) for value in candidate.shape)
    return collapsed


def compact_pad_axes(model: onnx.ModelProto) -> int:
    default_opset = next(
        (item.version for item in model.opset_import if item.domain in {"", "ai.onnx"}),
        0,
    )
    if default_opset < 18:
        return 0
    initializers = {item.name: item for item in model.graph.initializer}
    compacted = 0
    for node in model.graph.node:
        if node.op_type != "Pad" or len(node.input) < 2 or not node.input[1]:
            continue
        if len(node.input) >= 4 and node.input[3]:
            continue
        pads_tensor = initializers.get(node.input[1])
        if pads_tensor is None:
            continue
        pads = numpy_helper.to_array(pads_tensor)
        if pads.ndim != 1 or pads.size % 2:
            continue
        rank = pads.size // 2
        active = [
            axis
            for axis in range(rank)
            if int(pads[axis]) != 0 or int(pads[axis + rank]) != 0
        ]
        if not active or 3 * len(active) >= 2 * rank:
            continue
        compact = np.asarray(
            [*[pads[axis] for axis in active], *[pads[axis + rank] for axis in active]],
            dtype=pads.dtype,
        )
        pads_name = f"{pads_tensor.name}_micro_{compacted}"
        axes_name = f"pad_axes_micro_{'_'.join(map(str, active))}"
        existing_axes = next(
            (
                item
                for item in model.graph.initializer
                if item.name == axes_name
                and np.array_equal(numpy_helper.to_array(item), np.asarray(active, dtype=np.int64))
            ),
            None,
        )
        model.graph.initializer.append(numpy_helper.from_array(compact, name=pads_name))
        if existing_axes is None:
            model.graph.initializer.append(
                numpy_helper.from_array(np.asarray(active, dtype=np.int64), name=axes_name)
            )
        node.input[1] = pads_name
        while len(node.input) < 3:
            node.input.append("")
        if len(node.input) == 3:
            node.input.append(axes_name)
        else:
            node.input[3] = axes_name
        compacted += 1
    return compacted


def compact_constant_pad_axes(model: onnx.ModelProto) -> int:
    default_opset = next(
        (item.version for item in model.opset_import if item.domain in {"", "ai.onnx"}),
        0,
    )
    if default_opset < 18:
        return 0
    sources = _tensor_sources(model)
    initializer_names = {item.name for item in model.graph.initializer}
    compacted = 0
    for node in model.graph.node:
        if node.op_type != "Pad" or len(node.input) < 2 or not node.input[1]:
            continue
        if node.input[1] in initializer_names:
            continue
        if len(node.input) >= 4 and node.input[3]:
            continue
        pads = sources.get(node.input[1])
        if pads is None or pads.ndim != 1 or pads.size % 2:
            continue
        rank = pads.size // 2
        active = [
            axis
            for axis in range(rank)
            if int(pads[axis]) != 0 or int(pads[axis + rank]) != 0
        ]
        if not active or 3 * len(active) >= 2 * rank:
            continue
        compact = np.asarray(
            [*[pads[axis] for axis in active], *[pads[axis + rank] for axis in active]],
            dtype=pads.dtype,
        )
        pads_name = f"constant_pad_micro_{compacted}"
        axes_name = f"constant_pad_axes_micro_{compacted}"
        model.graph.initializer.extend(
            [
                numpy_helper.from_array(compact, name=pads_name),
                numpy_helper.from_array(np.asarray(active, dtype=np.int64), name=axes_name),
            ]
        )
        node.input[1] = pads_name
        while len(node.input) < 3:
            node.input.append("")
        if len(node.input) == 3:
            node.input.append(axes_name)
        else:
            node.input[3] = axes_name
        compacted += 1
    return compacted


def _zero_border(weight: np.ndarray) -> tuple[int, int, int, int]:
    if weight.ndim != 4:
        return 0, 0, 0, 0
    support = np.any(weight != 0, axis=(0, 1))
    if not support.any():
        return 0, 0, 0, 0
    rows = np.flatnonzero(support.any(axis=1))
    cols = np.flatnonzero(support.any(axis=0))
    return (
        int(rows[0]),
        int(weight.shape[2] - rows[-1] - 1),
        int(cols[0]),
        int(weight.shape[3] - cols[-1] - 1),
    )


def crop_conv_support(model: onnx.ModelProto) -> int:
    initializers = {item.name: item for item in model.graph.initializer}
    consumer_count: dict[str, int] = {}
    for graph_node in model.graph.node:
        for name in graph_node.input:
            if name:
                consumer_count[name] = consumer_count.get(name, 0) + 1
    cropped = 0
    for node in model.graph.node:
        if node.op_type not in {"Conv", "QLinearConv"}:
            continue
        auto_pad = _attribute(node, "auto_pad")
        if auto_pad is not None and auto_pad.s not in {b"", b"NOTSET"}:
            continue
        weight_index = 1 if node.op_type == "Conv" else 3
        if len(node.input) <= weight_index or node.input[weight_index] not in initializers:
            continue
        if consumer_count.get(node.input[weight_index], 0) != 1:
            continue
        tensor = initializers[node.input[weight_index]]
        weight = numpy_helper.to_array(tensor)
        top, bottom, left, right = _zero_border(weight)
        if top + bottom + left + right == 0:
            continue
        pads = _attribute(node, "pads")
        values = list(pads.ints) if pads is not None else [0, 0, 0, 0]
        dilation = _attribute(node, "dilations")
        dilation_values = list(dilation.ints) if dilation is not None else [1, 1]
        if len(values) != 4 or len(dilation_values) != 2:
            continue
        adjusted = [
            values[0] - top * dilation_values[0],
            values[1] - left * dilation_values[1],
            values[2] - bottom * dilation_values[0],
            values[3] - right * dilation_values[1],
        ]
        if any(value < 0 for value in adjusted):
            continue
        trimmed = weight[:, :, top : weight.shape[2] - bottom, left : weight.shape[3] - right]
        tensor.CopyFrom(numpy_helper.from_array(trimmed, name=tensor.name))
        if pads is None:
            pads = node.attribute.add()
            pads.name = "pads"
            pads.type = onnx.AttributeProto.INTS
        pads.ints[:] = adjusted
        kernel = _attribute(node, "kernel_shape")
        if kernel is not None:
            kernel.ints[:] = [trimmed.shape[2], trimmed.shape[3]]
        cropped += 1
    return cropped


def absorb_terminal_1x1_conv_bias(model: onnx.ModelProto) -> int:
    graph_inputs = {item.name for item in model.graph.input}
    graph_outputs = {item.name for item in model.graph.output}
    initializers = {item.name: item for item in model.graph.initializer}
    absorbed = 0
    for node in model.graph.node:
        if (
            node.op_type != "Conv"
            or len(node.input) != 3
            or len(node.output) != 1
            or node.input[0] not in graph_inputs
            or node.output[0] not in graph_outputs
        ):
            continue
        weight_tensor = initializers.get(node.input[1])
        bias_tensor = initializers.get(node.input[2])
        if weight_tensor is None or bias_tensor is None:
            continue
        weight = numpy_helper.to_array(weight_tensor)
        bias = numpy_helper.to_array(bias_tensor)
        if (
            weight.ndim != 4
            or weight.shape[2:] != (1, 1)
            or bias.ndim != 1
            or bias.shape[0] != weight.shape[0]
            or np.any(bias > 0)
        ):
            continue
        group = _attribute(node, "group")
        pads = _attribute(node, "pads")
        strides = _attribute(node, "strides")
        dilations = _attribute(node, "dilations")
        if group is not None and group.i != 1:
            continue
        if pads is not None and any(pads.ints):
            continue
        if strides is not None and list(strides.ints) != [1, 1]:
            continue
        if dilations is not None and list(dilations.ints) != [1, 1]:
            continue
        adjusted = weight + bias.reshape(-1, 1, 1, 1).astype(weight.dtype)
        weight_tensor.CopyFrom(numpy_helper.from_array(adjusted, name=weight_tensor.name))
        del node.input[2:]
        absorbed += int(bias.size)
    return absorbed


def sparsify_small_zero_initializers(model: onnx.ModelProto) -> int:
    supported_consumers = {"Conv", "QLinearConv", "Gemm", "MatMul"}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for name in node.input:
            if name:
                consumers.setdefault(name, []).append(node)
    kept: list[onnx.TensorProto] = []
    sparse_items: list[onnx.SparseTensorProto] = []
    saved = 0
    for item in model.graph.initializer:
        nodes = consumers.get(item.name, [])
        array = numpy_helper.to_array(item)
        if (
            not nodes
            or any(node.op_type not in supported_consumers for node in nodes)
            or array.ndim == 0
            or array.size <= 1
        ):
            kept.append(item)
            continue
        flat = array.reshape(-1)
        indices = np.flatnonzero(flat != 0).astype(np.int64)
        zero_count = int(flat.size - indices.size)
        if zero_count <= 0 or zero_count > 32 or indices.size == 0:
            kept.append(item)
            continue
        values = flat[indices]
        sparse_items.append(
            onnx.helper.make_sparse_tensor(
                numpy_helper.from_array(values, name=item.name),
                numpy_helper.from_array(indices, name=f"{item.name}_sparse_indices"),
                list(array.shape),
            )
        )
        saved += zero_count
    if saved:
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept)
        model.graph.sparse_initializer.extend(sparse_items)
    return saved


def eliminate_trivial_nodes(model: onnx.ModelProto) -> int:
    shapes = _shape_map(model)
    types = _type_map(model)
    graph_outputs = {item.name for item in model.graph.output}
    replacements: dict[str, str] = {}
    kept: list[onnx.NodeProto] = []

    def source(name: str) -> str:
        while name in replacements:
            name = replacements[name]
        return name

    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name:
                node.input[index] = source(name)
        removable = False
        if len(node.output) == 1 and node.output[0] not in graph_outputs and node.input:
            input_name = node.input[0]
            output_name = node.output[0]
            if node.op_type == "Identity":
                removable = True
            elif node.op_type == "Concat" and len([name for name in node.input if name]) == 1:
                removable = True
            elif node.op_type == "Transpose":
                perm = _attribute(node, "perm")
                rank = len(shapes.get(input_name, ()))
                values = list(perm.ints) if perm is not None else list(reversed(range(rank)))
                removable = bool(rank and values == list(range(rank)))
            elif node.op_type == "Cast":
                removable = types.get(input_name) is not None and types.get(input_name) == types.get(output_name)
            elif node.op_type == "Reshape":
                removable = shapes.get(input_name) is not None and shapes.get(input_name) == shapes.get(output_name)
            elif node.op_type == "Pad" and len(node.input) >= 2:
                initializer = next(
                    (item for item in model.graph.initializer if item.name == node.input[1]),
                    None,
                )
                removable = initializer is not None and not np.any(numpy_helper.to_array(initializer))
        if removable:
            replacements[node.output[0]] = source(node.input[0])
        else:
            kept.append(node)
    if not replacements:
        return 0
    for node in kept:
        for index, name in enumerate(node.input):
            if name:
                node.input[index] = source(name)
    removed = len(model.graph.node) - len(kept)
    del model.graph.node[:]
    model.graph.node.extend(kept)
    return removed


def remove_optional_default_inputs(model: onnx.ModelProto) -> int:
    sources = _tensor_sources(model)
    removed = 0
    for node in model.graph.node:
        if node.op_type in {"Conv", "Gemm"} and len(node.input) >= 3:
            value = sources.get(node.input[2])
            if value is not None and not np.any(value):
                del node.input[2:]
                removed += int(value.size or 1)
        elif node.op_type == "QLinearConv" and len(node.input) >= 9:
            value = sources.get(node.input[8])
            if value is not None and not np.any(value):
                del node.input[8:]
                removed += int(value.size or 1)
        elif node.op_type == "Pad" and len(node.input) >= 3 and node.input[2]:
            value = sources.get(node.input[2])
            if value is not None and value.size == 1 and not np.any(value):
                if len(node.input) == 3:
                    del node.input[2:]
                else:
                    node.input[2] = ""
                removed += 1
        elif node.op_type == "Slice" and len(node.input) >= 5 and node.input[4]:
            value = sources.get(node.input[4])
            if value is not None and value.size and np.all(value == 1):
                del node.input[4:]
                removed += int(value.size)
        elif node.op_type == "Dropout":
            if len(node.input) >= 3 and node.input[2]:
                training = sources.get(node.input[2])
                if training is not None and training.size == 1 and not bool(training.reshape(-1)[0]):
                    del node.input[2:]
                    removed += 1
            if len(node.input) >= 2 and node.input[1]:
                ratio = sources.get(node.input[1])
                if ratio is not None and ratio.size == 1 and float(ratio.reshape(-1)[0]) == 0.5:
                    del node.input[1:]
                    removed += 1
    return removed


def eliminate_neutral_elementwise(model: onnx.ModelProto) -> int:
    shapes = _shape_map(model)
    sources = _tensor_sources(model)
    graph_outputs = {item.name for item in model.graph.output}
    replacements: dict[str, str] = {}
    kept: list[onnx.NodeProto] = []

    def source(name: str) -> str:
        while name in replacements:
            name = replacements[name]
        return name

    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name:
                node.input[index] = source(name)
        replacement = ""
        if len(node.output) == 1 and node.output[0] not in graph_outputs:
            output_shape = shapes.get(node.output[0])
            values = [sources.get(name) for name in node.input]
            if node.op_type in {"Add", "Mul", "Or", "And", "Xor"} and len(node.input) == 2:
                for index in range(2):
                    value = values[index]
                    other = node.input[1 - index]
                    if value is None or shapes.get(other) != output_shape:
                        continue
                    neutral = (
                        not np.any(value)
                        if node.op_type in {"Add", "Or", "Xor"}
                        else np.all(value == 1)
                    )
                    if neutral:
                        replacement = other
                        break
            elif node.op_type in {"Sub", "Div"} and len(node.input) == 2:
                value = values[1]
                if value is not None and shapes.get(node.input[0]) == output_shape:
                    neutral = not np.any(value) if node.op_type == "Sub" else np.all(value == 1)
                    if neutral:
                        replacement = node.input[0]
            elif node.op_type == "Where" and len(node.input) == 3 and values[0] is not None:
                condition = values[0]
                selected = node.input[1] if np.all(condition) else node.input[2] if not np.any(condition) else ""
                if selected and shapes.get(selected) == output_shape:
                    replacement = selected
        if replacement:
            replacements[node.output[0]] = source(replacement)
        else:
            kept.append(node)
    if not replacements:
        return 0
    for node in kept:
        for index, name in enumerate(node.input):
            if name:
                node.input[index] = source(name)
    removed = len(model.graph.node) - len(kept)
    del model.graph.node[:]
    model.graph.node.extend(kept)
    return removed


def replace_nonnegative_scalar_offsets_with_shrink(model: onnx.ModelProto) -> int:
    sources = _tensor_sources(model)
    producers = {
        output: node
        for node in model.graph.node
        for output in node.output
        if output
    }
    changed = 0
    for node in model.graph.node:
        dynamic = ""
        bias = 0.0
        if node.op_type == "Add" and len(node.input) == 2:
            for index in range(2):
                value = sources.get(node.input[index])
                if value is not None and value.size == 1 and float(value.reshape(-1)[0]) < 0:
                    dynamic = node.input[1 - index]
                    bias = -float(value.reshape(-1)[0])
                    break
        elif node.op_type == "Sub" and len(node.input) == 2:
            value = sources.get(node.input[1])
            if value is not None and value.size == 1 and float(value.reshape(-1)[0]) > 0:
                dynamic = node.input[0]
                bias = float(value.reshape(-1)[0])
        if not dynamic or bias <= 0:
            continue
        producer = producers.get(dynamic)
        if producer is None or producer.op_type != "Cast" or not producer.input:
            continue
        origin = producers.get(producer.input[0])
        if origin is None or origin.op_type not in {"ReduceL1", "ReduceL2", "ReduceSumSquare"}:
            continue
        node.op_type = "Shrink"
        del node.input[:]
        node.input.append(dynamic)
        del node.attribute[:]
        node.attribute.extend(
            [
                onnx.helper.make_attribute("bias", bias),
                onnx.helper.make_attribute("lambd", 0.0),
            ]
        )
        changed += 1
    return changed


def compact_unit_reduction_axes(model: onnx.ModelProto) -> int:
    supported = {
        "ReduceL1",
        "ReduceL2",
        "ReduceLogSum",
        "ReduceLogSumExp",
        "ReduceMax",
        "ReduceMean",
        "ReduceMin",
        "ReduceProd",
        "ReduceSum",
        "ReduceSumSquare",
    }
    shapes = _shape_map(model)
    sources = _tensor_sources(model)
    changed = 0
    for node in model.graph.node:
        if node.op_type not in supported or len(node.input) < 2 or not node.input[1]:
            continue
        input_shape = shapes.get(node.input[0])
        axes = sources.get(node.input[1])
        if input_shape is None or axes is None or axes.ndim > 1 or axes.size <= 1:
            continue
        rank = len(input_shape)
        normalized = [int(axis) % rank for axis in axes.reshape(-1)]
        retained = [axis for axis, normalized_axis in zip(axes.reshape(-1), normalized) if input_shape[normalized_axis] != 1]
        if not retained or len(retained) == axes.size:
            continue
        compact = np.asarray(retained, dtype=axes.dtype)
        if _replace_source_array(model, node.input[1], compact):
            changed += int(axes.size - compact.size)
            sources[node.input[1]] = compact
    return changed


def _parameter_elements(model: onnx.ModelProto) -> int:
    total = sum(int(np.prod(item.dims or [1])) for item in model.graph.initializer)
    for node in model.graph.node:
        value = _constant_array(node)
        if value is not None:
            total += int(value.size or 1)
    return total


def downgrade_to_opset12(model: onnx.ModelProto) -> int:
    current = next(
        (item.version for item in model.opset_import if item.domain in {"", "ai.onnx"}),
        0,
    )
    if current <= 12:
        return 0
    before = _parameter_elements(model)
    try:
        converted = version_converter.convert_version(model, 12)
    except Exception:
        return 0
    after = _parameter_elements(converted)
    if after >= before:
        return 0
    model.CopyFrom(converted)
    return before - after


TRANSFORMS = {
    "pad_axes": compact_pad_axes,
    "constant_pad_axes": compact_constant_pad_axes,
    "conv_crop": crop_conv_support,
    "terminal_conv_bias_absorb": absorb_terminal_1x1_conv_bias,
    "small_sparse_initializer": sparsify_small_zero_initializers,
    "broadcast_init": collapse_broadcast_initializers,
    "trivial_nodes": eliminate_trivial_nodes,
    "optional_defaults": remove_optional_default_inputs,
    "neutral_elementwise": eliminate_neutral_elementwise,
    "nonnegative_offset_shrink": replace_nonnegative_scalar_offsets_with_shrink,
    "unit_reduction_axes": compact_unit_reduction_axes,
    "constant_dedup": deduplicate_constant_tensors,
    "opset12_controls": downgrade_to_opset12,
    "init_cleanup": lambda model: deduplicate_initializers(model) + prune_initializers(model),
}


def apply_transforms(source: onnx.ModelProto, names: list[str]) -> tuple[onnx.ModelProto, list[str]]:
    model = deepcopy(source)
    changes: list[str] = []
    for name in names:
        count = TRANSFORMS[name](model)
        if count:
            changes.append(f"{name}:{count}")
    cleanup = (
        deduplicate_constant_tensors(model)
        + deduplicate_initializers(model)
        + prune_initializers(model)
    )
    if cleanup:
        changes.append(f"final_init_cleanup:{cleanup}")
    return model, changes


def graph_stats(model: onnx.ModelProto) -> dict[str, int]:
    return {
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "initializer_elements": sum(int(np.prod(item.dims or [1])) for item in model.graph.initializer),
    }


def run_task(job: tuple[str, str, str]) -> dict[str, object]:
    task, parent_dir_raw, artifact_dir_raw = job
    parent_path = Path(parent_dir_raw) / f"{task}.onnx"
    artifact_dir = Path(artifact_dir_raw)
    row: dict[str, object] = {
        "task": task,
        "parent_path": str(parent_path),
        "candidate_path": "",
        "attempted": True,
        "source_policy": "parent_only_no_archive",
    }
    if not parent_path.exists():
        row["status"] = "missing_parent"
        return row

    source = onnx.load(str(parent_path))
    row.update({f"parent_{key}": value for key, value in graph_stats(source).items()})
    variants: list[tuple[str, onnx.ModelProto, list[str]]] = []
    all_names = list(TRANSFORMS)
    combined, combined_changes = apply_transforms(source, all_names)
    if combined_changes:
        variants.append(("combined", combined, combined_changes))
    for name in all_names:
        model, changes = apply_transforms(source, [name])
        if changes:
            variants.append((name, model, changes))

    unique_variants: list[tuple[str, onnx.ModelProto, list[str]]] = []
    hashes: set[str] = set()
    for name, model, changes in variants:
        digest = hashlib.sha256(model.SerializeToString()).hexdigest()
        if digest not in hashes:
            hashes.add(digest)
            unique_variants.append((name, model, changes))
    row["variant_count"] = len(unique_variants)
    row["opportunities"] = ";".join(
        f"{name}={'|'.join(changes)}" for name, _, changes in unique_variants
    )
    if not unique_variants:
        row["status"] = "attempted_no_structural_opportunity"
        return row

    sys.path.insert(0, str(HERE.parent))
    from c_score_common import score_onnx

    parent_score = score_onnx(task, parent_path, True)
    row.update({f"parent_score_{key}": value for key, value in asdict(parent_score).items()})
    best: tuple[int, str, Path, object, list[str], dict[str, int]] | None = None
    errors: list[str] = []
    for index, (name, model, changes) in enumerate(unique_variants):
        variant_path = artifact_dir / task / f"{name}.onnx"
        variant_path.parent.mkdir(parents=True, exist_ok=True)
        model.producer_name = "ngc_cd_parent_micro_sweep"
        try:
            onnx.checker.check_model(model, full_check=True)
            onnx.save(model, str(variant_path))
        except Exception as exc:
            errors.append(f"{name}:checker:{type(exc).__name__}:{exc}")
            continue
        score = score_onnx(task, variant_path, True)
        if not score.ok:
            errors.append(
                f"{name}:validation:{score.examples_passed}/{score.examples_checked}:{score.error}"
            )
            continue
        if score.cost is None:
            errors.append(f"{name}:missing_cost")
            continue
        candidate = (
            score.cost,
            name,
            variant_path,
            score,
            changes,
            graph_stats(model),
        )
        if best is None or candidate[0] < best[0]:
            best = candidate

    row["errors"] = " || ".join(errors)
    if best is None:
        row["status"] = "all_variants_failed"
        return row
    cost, name, path, score, changes, stats = best
    row.update({f"candidate_{key}": value for key, value in asdict(score).items()})
    row.update({f"candidate_graph_{key}": value for key, value in stats.items()})
    row["selected_variant"] = name
    row["selected_changes"] = ";".join(changes)
    row["candidate_path"] = str(path)
    row["delta_cost"] = (
        parent_score.cost - cost if parent_score.cost is not None else ""
    )
    row["delta_points"] = (
        score.points - parent_score.points
        if score.points is not None and parent_score.points is not None
        else ""
    )
    row["accepted"] = bool(parent_score.ok and parent_score.cost is not None and cost < parent_score.cost)
    row["status"] = "accepted" if row["accepted"] else "valid_no_cost_gain"

    selected_path = artifact_dir / f"{task}.onnx"
    if row["accepted"]:
        shutil.copy2(path, selected_path)
        row["candidate_path"] = str(selected_path)
    return row


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--scope", choices=["cd", "archive", "all"], default="cd")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--parent-dir", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    if args.scope == "cd":
        with args.assignments.open(newline="", encoding="utf-8-sig") as handle:
            assignments = list(csv.DictReader(handle))
        tasks = sorted(row["task"] for row in assignments if row.get("owner") in {"C", "D"})
        expected = 134
    elif args.scope == "archive":
        tasks = sorted(path.stem for path in args.archive_dir.glob("task*.onnx"))
        expected = 399
    else:
        tasks = [f"task{index:03d}" for index in range(1, 401)]
        expected = 400
    if args.tasks:
        tasks = sorted({item.strip() for item in args.tasks.split(",") if item.strip()})
        expected = len(tasks)
    if len(tasks) != expected:
        raise RuntimeError(f"expected {expected} {args.scope} tasks, found {len(tasks)}")

    rows: list[dict[str, object]] = []
    jobs = [(task, str(args.parent_dir.resolve()), str(args.artifact_dir.resolve())) for task in tasks]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_task, job): job[0] for job in jobs}
        for index, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "task": task,
                    "attempted": True,
                    "source_policy": "parent_only_no_archive",
                    "status": "worker_failed",
                    "errors": f"{type(exc).__name__}:{exc}",
                }
            rows.append(row)
            rows.sort(key=lambda item: str(item["task"]))
            write_rows(args.output, rows)
            print(
                json.dumps(
                    {
                        "completed": index,
                        "task": task,
                        "status": row.get("status"),
                        "delta_cost": row.get("delta_cost", ""),
                    }
                ),
                flush=True,
            )

    summary = {
        "tasks": len(rows),
        "attempted": sum(bool(row.get("attempted")) for row in rows),
        "accepted": sum(bool(row.get("accepted")) for row in rows),
        "total_delta_cost": sum(int(row.get("delta_cost") or 0) for row in rows if row.get("accepted")),
        "total_delta_points": sum(float(row.get("delta_points") or 0) for row in rows if row.get("accepted")),
        "status_counts": {
            status: sum(row.get("status") == status for row in rows)
            for status in sorted({str(row.get("status")) for row in rows})
        },
        "source_policy": "parent_only_no_archive",
        "scope": args.scope,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
