from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def _shape_map(model: onnx.ModelProto) -> dict[str, tuple[int, ...]]:
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=False, data_prop=True
    )
    result: dict[str, tuple[int, ...]] = {}
    for item in [
        *inferred.graph.input,
        *inferred.graph.value_info,
        *inferred.graph.output,
    ]:
        dims = item.type.tensor_type.shape.dim
        if all(dim.HasField("dim_value") for dim in dims):
            result[item.name] = tuple(int(dim.dim_value) for dim in dims)
    for item in inferred.graph.initializer:
        result[item.name] = tuple(int(dim) for dim in item.dims)
    return result


def _attribute(node: onnx.NodeProto, name: str) -> onnx.AttributeProto | None:
    return next((item for item in node.attribute if item.name == name), None)


def concat_constant_to_pad(model: onnx.ModelProto) -> int:
    shapes = _shape_map(model)
    initializers = {
        item.name: item for item in model.graph.initializer
    }
    values = {
        name: numpy_helper.to_array(item) for name, item in initializers.items()
    }
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for name in node.input:
            if name:
                consumers.setdefault(name, []).append(node)
    existing_names = {
        name
        for node in model.graph.node
        for name in [*node.input, *node.output]
        if name
    } | set(initializers)

    converted = 0
    for node in model.graph.node:
        if node.op_type != "Concat" or len(node.input) < 2:
            continue
        dynamic_positions = [
            index for index, name in enumerate(node.input) if name not in values
        ]
        if len(dynamic_positions) != 1:
            continue
        dynamic_index = dynamic_positions[0]
        dynamic_name = node.input[dynamic_index]
        dynamic_shape = shapes.get(dynamic_name)
        output_shape = shapes.get(node.output[0]) if node.output else None
        axis_attribute = _attribute(node, "axis")
        if dynamic_shape is None or output_shape is None or axis_attribute is None:
            continue
        rank = len(dynamic_shape)
        axis = int(axis_attribute.i) % rank
        if len(output_shape) != rank:
            continue

        constant_names = [
            name for index, name in enumerate(node.input) if index != dynamic_index
        ]
        arrays = [values[name] for name in constant_names]
        flattened = np.concatenate([array.reshape(-1) for array in arrays])
        if flattened.size == 0 or not np.all(flattened == flattened[0]):
            continue
        valid_shapes = True
        for array in arrays:
            if array.ndim != rank:
                valid_shapes = False
                break
            for dim, (constant_size, dynamic_size) in enumerate(
                zip(array.shape, dynamic_shape)
            ):
                if dim != axis and int(constant_size) != int(dynamic_size):
                    valid_shapes = False
                    break
        if not valid_shapes:
            continue

        before = sum(int(array.shape[axis]) for array in arrays[:dynamic_index])
        after = sum(int(array.shape[axis]) for array in arrays[dynamic_index:])
        if output_shape[axis] != before + dynamic_shape[axis] + after:
            continue
        uniquely_removed = {
            name
            for name in constant_names
            if all(consumer is node for consumer in consumers.get(name, []))
        }
        saved = sum(int(values[name].size) for name in uniquely_removed)
        added = 2 * rank + 1
        if saved <= added:
            continue

        base = f"{node.output[0]}_concat_pad"
        pads_name = f"{base}_pads"
        fill_name = f"{base}_fill"
        suffix = 0
        while pads_name in existing_names or fill_name in existing_names:
            suffix += 1
            pads_name = f"{base}_pads_{suffix}"
            fill_name = f"{base}_fill_{suffix}"
        existing_names.update({pads_name, fill_name})
        pads = np.zeros(2 * rank, dtype=np.int64)
        pads[axis] = before
        pads[axis + rank] = after
        fill = np.asarray(flattened[0], dtype=arrays[0].dtype)
        model.graph.initializer.extend(
            [
                numpy_helper.from_array(pads, name=pads_name),
                numpy_helper.from_array(fill, name=fill_name),
            ]
        )
        node.op_type = "Pad"
        del node.input[:]
        node.input.extend([dynamic_name, pads_name, fill_name])
        del node.attribute[:]
        converted += 1

    if not converted:
        return 0
    used = {name for node in model.graph.node for name in node.input if name}
    kept = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return converted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_model", type=Path)
    parser.add_argument("output_model", type=Path)
    args = parser.parse_args()
    model = onnx.load(args.input_model)
    converted = concat_constant_to_pad(model)
    if not converted:
        raise SystemExit("no beneficial constant Concat -> Pad conversion")
    model.producer_name = "ngc_concat_constant_to_pad"
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output_model)
    print(f"converted={converted} output={args.output_model}")


if __name__ == "__main__":
    main()
