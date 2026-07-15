from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import onnx


SUPPORTED = {"Gather", "GatherElements", "GatherND", "Slice"}


def _unique_name(model: onnx.ModelProto, base: str) -> str:
    names = {
        name
        for node in model.graph.node
        for name in (*node.input, *node.output)
        if name
    }
    names.update(item.name for item in model.graph.initializer)
    names.update(item.name for item in model.graph.input)
    names.update(item.name for item in model.graph.output)
    if base not in names:
        return base
    index = 1
    while f"{base}_{index}" in names:
        index += 1
    return f"{base}_{index}"


def push(model: onnx.ModelProto) -> int:
    """Move a single-use Cast after a data-selection operation.

    Gather and Slice only select/reorder values, so applying Cast after them is
    exactly equivalent.  The rewrite is useful when the selection is smaller
    than its input because the Cast materializes fewer elements.
    """

    nodes = list(model.graph.node)
    uses = Counter(name for node in nodes for name in node.input if name)
    graph_outputs = {item.name for item in model.graph.output}
    producer = {
        output: index
        for index, node in enumerate(nodes)
        for output in node.output
        if output
    }
    replacements: dict[int, tuple[onnx.NodeProto, onnx.NodeProto]] = {}
    removed: set[int] = set()
    changed = 0

    for selection_index, selection in enumerate(nodes):
        if (
            selection.op_type not in SUPPORTED
            or not selection.input
            or not selection.output
            or selection_index in removed
        ):
            continue
        cast_index = producer.get(selection.input[0])
        if cast_index is None or cast_index >= selection_index or cast_index in removed:
            continue
        cast = nodes[cast_index]
        if (
            cast.op_type != "Cast"
            or len(cast.input) != 1
            or len(cast.output) != 1
            or uses[cast.output[0]] != 1
            or cast.output[0] in graph_outputs
            or len(selection.output) != 1
        ):
            continue

        selected_name = _unique_name(
            model,
            f"{selection.output[0]}__before_cast",
        )
        moved_selection = onnx.NodeProto()
        moved_selection.CopyFrom(selection)
        moved_selection.input[0] = cast.input[0]
        moved_selection.output[0] = selected_name

        moved_cast = onnx.NodeProto()
        moved_cast.CopyFrom(cast)
        moved_cast.input[0] = selected_name
        moved_cast.output[0] = selection.output[0]

        replacements[selection_index] = (moved_selection, moved_cast)
        removed.add(cast_index)
        changed += 1

    if not changed:
        return 0

    result: list[onnx.NodeProto] = []
    for index, node in enumerate(nodes):
        if index in removed:
            continue
        replacement = replacements.get(index)
        if replacement is None:
            result.append(node)
        else:
            result.extend(replacement)
    del model.graph.node[:]
    model.graph.node.extend(result)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = onnx.load(args.input)
    count = push(model)
    if count <= 0:
        raise SystemExit(2)
    model.producer_name = "ngc_push_cast_through_selection"
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)


if __name__ == "__main__":
    main()
