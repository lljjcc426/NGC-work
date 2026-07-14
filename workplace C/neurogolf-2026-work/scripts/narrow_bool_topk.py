from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _type_map(model: onnx.ModelProto) -> dict[str, int]:
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=False, data_prop=True
    )
    result: dict[str, int] = {}
    for item in [
        *inferred.graph.input,
        *inferred.graph.value_info,
        *inferred.graph.output,
    ]:
        tensor_type = item.type.tensor_type
        if tensor_type.elem_type:
            result[item.name] = int(tensor_type.elem_type)
    return result


def _cast_target(node: onnx.NodeProto) -> int | None:
    return next(
        (int(attribute.i) for attribute in node.attribute if attribute.name == "to"),
        None,
    )


def _constant_map(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }


def narrow_bool_topk(model: onnx.ModelProto) -> int:
    nodes = list(model.graph.node)
    types = _type_map(model)
    constants = _constant_map(model)
    producers = {
        output: index
        for index, node in enumerate(nodes)
        for output in node.output
        if output
    }
    consumers: dict[str, list[int]] = {}
    for index, node in enumerate(nodes):
        for name in node.input:
            if name:
                consumers.setdefault(name, []).append(index)

    replacements: dict[int, onnx.NodeProto] = {}
    narrowed = 0
    for topk_index, topk in enumerate(nodes):
        if topk.op_type != "TopK" or len(topk.input) < 1 or len(topk.output) != 2:
            continue
        cast_index = producers.get(topk.input[0])
        if cast_index is None:
            continue
        cast = nodes[cast_index]
        if (
            cast.op_type != "Cast"
            or len(cast.input) != 1
            or len(cast.output) != 1
            or types.get(cast.input[0]) != TensorProto.BOOL
            or _cast_target(cast) not in {TensorProto.FLOAT16, TensorProto.FLOAT}
            or len(consumers.get(cast.output[0], [])) != 1
        ):
            continue
        value_consumers = consumers.get(topk.output[0], [])
        if len(value_consumers) != 1:
            continue
        compare_index = value_consumers[0]
        compare = nodes[compare_index]
        replace_compare = False
        if compare.op_type == "Cast" and _cast_target(compare) == TensorProto.BOOL:
            pass
        elif compare.op_type == "Greater" and len(compare.input) == 2:
            if compare.input[0] == topk.output[0]:
                zero_name = compare.input[1]
            elif compare.input[1] == topk.output[0]:
                zero_name = compare.input[0]
            else:
                continue
            zero = constants.get(zero_name)
            if zero is None or zero.size != 1 or np.any(zero):
                continue
            replace_compare = True
        else:
            continue

        for attribute in cast.attribute:
            if attribute.name == "to":
                attribute.i = TensorProto.INT8
                break
        for value_info in model.graph.value_info:
            if value_info.name in {cast.output[0], topk.output[0]}:
                value_info.type.tensor_type.elem_type = TensorProto.INT8
        if replace_compare:
            replacements[compare_index] = helper.make_node(
                "Cast",
                [topk.output[0]],
                list(compare.output),
                to=TensorProto.BOOL,
                name=compare.name,
            )
        narrowed += 1

    if not narrowed:
        return 0
    rewritten = [replacements.get(index, node) for index, node in enumerate(nodes)]
    del model.graph.node[:]
    model.graph.node.extend(rewritten)

    used = {name for node in model.graph.node for name in node.input if name}
    kept = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return narrowed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_model", type=Path)
    parser.add_argument("output_model", type=Path)
    args = parser.parse_args()

    model = onnx.load(args.input_model)
    narrowed = narrow_bool_topk(model)
    if not narrowed:
        raise SystemExit("no bool Cast -> TopK -> Greater(0) chain found")
    model.producer_name = "ngc_narrow_bool_topk"
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output_model)
    print(f"narrowed={narrowed} output={args.output_model}")


if __name__ == "__main__":
    main()
