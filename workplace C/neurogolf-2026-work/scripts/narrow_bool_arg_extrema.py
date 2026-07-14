from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto


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


def _cast_target(node: onnx.NodeProto) -> int | None:
    return next(
        (int(attribute.i) for attribute in node.attribute if attribute.name == "to"),
        None,
    )


def narrow_bool_arg_extrema(model: onnx.ModelProto) -> int:
    types = _type_map(model)
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for name in node.input:
            if name:
                consumers.setdefault(name, []).append(node)

    narrowed = 0
    for cast in model.graph.node:
        if (
            cast.op_type != "Cast"
            or len(cast.input) != 1
            or len(cast.output) != 1
            or types.get(cast.input[0]) != TensorProto.BOOL
            or _cast_target(cast) == TensorProto.UINT8
        ):
            continue
        users = consumers.get(cast.output[0], [])
        if not users or any(user.op_type not in {"ArgMax", "ArgMin"} for user in users):
            continue
        for attribute in cast.attribute:
            if attribute.name == "to":
                attribute.i = TensorProto.UINT8
                break
        for value_info in model.graph.value_info:
            if value_info.name == cast.output[0]:
                value_info.type.tensor_type.elem_type = TensorProto.UINT8
        narrowed += 1
    return narrowed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Narrow Boolean masks used only by ArgMax/ArgMin to uint8."
    )
    parser.add_argument("input_model", type=Path)
    parser.add_argument("output_model", type=Path)
    args = parser.parse_args()

    model = onnx.load(args.input_model)
    narrowed = narrow_bool_arg_extrema(model)
    if not narrowed:
        raise SystemExit("no exclusive Bool Cast -> ArgMax/ArgMin chain found")
    model.producer_name = "ngc_narrow_bool_arg_extrema"
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output_model)
    print(f"narrowed={narrowed} output={args.output_model}")


if __name__ == "__main__":
    main()
