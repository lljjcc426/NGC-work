from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


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


def narrow_bool_extrema_family(model: onnx.ModelProto) -> int:
    nodes = list(model.graph.node)
    types = _type_map(model)
    consumers: dict[str, list[int]] = {}
    for index, node in enumerate(nodes):
        for name in node.input:
            if name:
                consumers.setdefault(name, []).append(index)

    inserted_after: dict[int, list[onnx.NodeProto]] = {}
    narrowed = 0
    for cast in nodes:
        if (
            cast.op_type != "Cast"
            or len(cast.input) != 1
            or len(cast.output) != 1
            or types.get(cast.input[0]) != TensorProto.BOOL
        ):
            continue
        original_type = _cast_target(cast)
        if original_type == TensorProto.UINT8:
            continue
        user_indexes = consumers.get(cast.output[0], [])
        users = [nodes[index] for index in user_indexes]
        if not users or any(
            user.op_type not in {"ArgMax", "ArgMin", "ReduceMax", "ReduceMin"}
            for user in users
        ):
            continue

        for attribute in cast.attribute:
            if attribute.name == "to":
                attribute.i = TensorProto.UINT8
                break
        for value_info in model.graph.value_info:
            if value_info.name == cast.output[0]:
                value_info.type.tensor_type.elem_type = TensorProto.UINT8

        for user_index, user in zip(user_indexes, users):
            if user.op_type not in {"ReduceMax", "ReduceMin"}:
                continue
            original_output = user.output[0]
            compact_output = f"{original_output}_u8_extrema"
            user.output[0] = compact_output
            inserted_after.setdefault(user_index, []).append(
                helper.make_node(
                    "Cast",
                    [compact_output],
                    [original_output],
                    to=original_type,
                    name=f"{original_output}_restore_cast",
                )
            )
        narrowed += 1

    if not narrowed:
        return 0
    rewritten: list[onnx.NodeProto] = []
    for index, node in enumerate(nodes):
        rewritten.append(node)
        rewritten.extend(inserted_after.get(index, []))
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    return narrowed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Narrow Boolean extrema families to uint8."
    )
    parser.add_argument("input_model", type=Path)
    parser.add_argument("output_model", type=Path)
    args = parser.parse_args()

    model = onnx.load(args.input_model)
    narrowed = narrow_bool_extrema_family(model)
    if not narrowed:
        raise SystemExit("no Bool Cast extrema family found")
    model.producer_name = "ngc_narrow_bool_extrema_family"
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output_model)
    print(f"narrowed={narrowed} output={args.output_model}")


if __name__ == "__main__":
    main()
