from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper


FOLDABLE = {
    "Abs",
    "Add",
    "And",
    "ArgMax",
    "ArgMin",
    "Cast",
    "Ceil",
    "Clip",
    "Concat",
    "ConstantOfShape",
    "CumSum",
    "Div",
    "Equal",
    "Expand",
    "Flatten",
    "Floor",
    "Gather",
    "GatherElements",
    "GatherND",
    "Greater",
    "GreaterOrEqual",
    "Identity",
    "Less",
    "LessOrEqual",
    "MatMul",
    "Max",
    "Min",
    "Mod",
    "Mul",
    "Neg",
    "Not",
    "OneHot",
    "Or",
    "Pad",
    "Range",
    "ReduceMax",
    "ReduceMin",
    "ReduceSum",
    "Reshape",
    "Shape",
    "Sign",
    "Size",
    "Slice",
    "Squeeze",
    "Sub",
    "Tile",
    "Transpose",
    "Trilu",
    "Unsqueeze",
    "Where",
    "Xor",
}


def _evaluate(
    model: onnx.ModelProto,
    node: onnx.NodeProto,
    initializers: dict[str, onnx.TensorProto],
) -> list[np.ndarray]:
    inputs = [
        copy.deepcopy(initializers[name])
        for name in node.input
        if name
    ]
    outputs = [
        helper.make_tensor_value_info(name, onnx.TensorProto.UNDEFINED, None)
        for name in node.output
        if name
    ]
    graph = helper.make_graph(
        [copy.deepcopy(node)],
        "ngc_static_fold_probe",
        [],
        outputs,
        initializer=inputs,
    )
    probe = helper.make_model(
        graph,
        opset_imports=[copy.deepcopy(item) for item in model.opset_import],
    )
    probe.ir_version = model.ir_version
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 4
    session = ort.InferenceSession(
        probe.SerializeToString(),
        options,
        providers=["CPUExecutionProvider"],
    )
    return [np.asarray(value) for value in session.run(None, {})]


def _prune_initializers(model: onnx.ModelProto) -> int:
    used = {
        name
        for node in model.graph.node
        for name in node.input
        if name
    }
    used.update(item.name for item in model.graph.output)
    kept = [item for item in model.graph.initializer if item.name in used]
    removed = len(model.graph.initializer) - len(kept)
    if removed:
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept)
    return removed


def fold(model: onnx.ModelProto) -> int:
    graph_outputs = {item.name for item in model.graph.output}
    folded = 0
    while True:
        initializers = {item.name: item for item in model.graph.initializer}
        match: tuple[int, onnx.NodeProto, list[np.ndarray]] | None = None
        for index, node in enumerate(model.graph.node):
            if (
                node.op_type not in FOLDABLE
                or not node.input
                or any(output in graph_outputs for output in node.output)
                or not all(name in initializers for name in node.input if name)
            ):
                continue
            try:
                values = _evaluate(model, node, initializers)
            except Exception:
                continue
            if len(values) != len([name for name in node.output if name]):
                continue
            if any(value.dtype == object or value.size > 1_000_000 for value in values):
                continue
            match = (index, node, values)
            break
        if match is None:
            break

        index, node, values = match
        output_names = [name for name in node.output if name]
        del model.graph.node[index]
        model.graph.initializer.extend(
            numpy_helper.from_array(value, name=name)
            for name, value in zip(output_names, values)
        )
        folded += 1

    if folded:
        _prune_initializers(model)
    return folded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = onnx.load(args.input)
    count = fold(model)
    if count <= 0:
        raise SystemExit(2)
    model.producer_name = "ngc_fold_static_subgraphs"
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)


if __name__ == "__main__":
    main()
