from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


_INTEGER_TYPES = {
    TensorProto.UINT8,
    TensorProto.UINT16,
    TensorProto.UINT32,
    TensorProto.UINT64,
    TensorProto.INT8,
    TensorProto.INT16,
    TensorProto.INT32,
    TensorProto.INT64,
}


def _type_map(model: onnx.ModelProto) -> dict[str, int]:
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=False, data_prop=True
    )
    return {
        item.name: int(item.type.tensor_type.elem_type)
        for item in [
            *inferred.graph.input,
            *inferred.graph.value_info,
            *inferred.graph.output,
        ]
        if item.type.tensor_type.elem_type
    }


def _affine(
    node: onnx.NodeProto,
    values: dict[str, np.ndarray],
) -> tuple[str, int, np.ndarray] | None:
    if node.op_type not in {"Add", "Sub"} or len(node.input) != 2:
        return None
    constant_positions = [i for i, name in enumerate(node.input) if name in values]
    if len(constant_positions) != 1:
        return None
    constant_position = constant_positions[0]
    dynamic_position = 1 - constant_position
    dynamic = node.input[dynamic_position]
    constant = values[node.input[constant_position]]
    zero = np.zeros_like(constant)
    if node.op_type == "Add":
        return dynamic, 1, constant
    if dynamic_position == 0:
        return dynamic, 1, np.subtract(zero, constant, dtype=constant.dtype)
    return dynamic, -1, constant


def fold_integer_affine_constants(model: onnx.ModelProto) -> int:
    types = _type_map(model)
    graph_outputs = {item.name for item in model.graph.output}
    folded = 0

    while True:
        initializers = {item.name: item for item in model.graph.initializer}
        values = {
            name: numpy_helper.to_array(item) for name, item in initializers.items()
        }
        producers = {
            output: node
            for node in model.graph.node
            for output in node.output
            if output
        }
        consumers: dict[str, list[onnx.NodeProto]] = {}
        for node in model.graph.node:
            for name in node.input:
                if name:
                    consumers.setdefault(name, []).append(node)

        changed = False
        for outer in model.graph.node:
            if not outer.output or types.get(outer.output[0]) not in _INTEGER_TYPES:
                continue
            outer_affine = _affine(outer, values)
            if outer_affine is None:
                continue
            inner_output, outer_coefficient, outer_bias = outer_affine
            inner = producers.get(inner_output)
            if (
                inner is None
                or inner_output in graph_outputs
                or consumers.get(inner_output) != [outer]
            ):
                continue
            inner_affine = _affine(inner, values)
            if inner_affine is None:
                continue
            base, inner_coefficient, inner_bias = inner_affine
            if inner_bias.dtype != outer_bias.dtype:
                continue

            try:
                if outer_coefficient == 1:
                    bias = np.add(inner_bias, outer_bias, dtype=inner_bias.dtype)
                else:
                    bias = np.add(
                        np.subtract(
                            np.zeros_like(inner_bias),
                            inner_bias,
                            dtype=inner_bias.dtype,
                        ),
                        outer_bias,
                        dtype=inner_bias.dtype,
                    )
            except ValueError:
                continue
            coefficient = outer_coefficient * inner_coefficient

            name_base = f"{outer.output[0]}_folded_bias"
            existing = {
                item.name for item in model.graph.initializer
            } | {
                name
                for node in model.graph.node
                for name in [*node.input, *node.output]
                if name
            }
            bias_name = name_base
            suffix = 0
            while bias_name in existing:
                suffix += 1
                bias_name = f"{name_base}_{suffix}"
            model.graph.initializer.append(
                numpy_helper.from_array(np.asarray(bias), name=bias_name)
            )

            outer.op_type = "Add" if coefficient == 1 else "Sub"
            del outer.input[:]
            if coefficient == 1:
                outer.input.extend([base, bias_name])
            else:
                outer.input.extend([bias_name, base])
            model.graph.node.remove(inner)

            used = {
                name for node in model.graph.node for name in node.input if name
            }
            kept = [item for item in model.graph.initializer if item.name in used]
            del model.graph.initializer[:]
            model.graph.initializer.extend(kept)
            folded += 1
            changed = True
            break
        if not changed:
            return folded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fold exclusive integer Add/Sub constant chains."
    )
    parser.add_argument("input_model", type=Path)
    parser.add_argument("output_model", type=Path)
    args = parser.parse_args()

    model = onnx.load(args.input_model)
    folded = fold_integer_affine_constants(model)
    if not folded:
        raise SystemExit("no integer affine constant chain found")
    model.producer_name = "ngc_fold_integer_affine_constants"
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output_model)
    print(f"folded={folded} output={args.output_model}")


if __name__ == "__main__":
    main()
