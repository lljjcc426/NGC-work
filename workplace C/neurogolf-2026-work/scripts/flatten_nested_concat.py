from __future__ import annotations

import argparse
from pathlib import Path

import onnx


def _axis(node: onnx.NodeProto) -> int | None:
    attribute = next((item for item in node.attribute if item.name == "axis"), None)
    return int(attribute.i) if attribute is not None else None


def flatten_nested_concat(model: onnx.ModelProto) -> int:
    flattened = 0
    graph_outputs = {item.name for item in model.graph.output}

    while True:
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
            if outer.op_type != "Concat":
                continue
            outer_axis = _axis(outer)
            if outer_axis is None:
                continue
            for input_index, name in enumerate(list(outer.input)):
                inner = producers.get(name)
                if inner is None or inner.op_type != "Concat":
                    continue
                if _axis(inner) != outer_axis:
                    continue
                if name in graph_outputs or consumers.get(name) != [outer]:
                    continue

                expanded = [
                    *outer.input[:input_index],
                    *inner.input,
                    *outer.input[input_index + 1 :],
                ]
                del outer.input[:]
                outer.input.extend(expanded)
                model.graph.node.remove(inner)
                flattened += 1
                changed = True
                break
            if changed:
                break
        if not changed:
            return flattened


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flatten exclusive nested Concat nodes with the same axis."
    )
    parser.add_argument("input_model", type=Path)
    parser.add_argument("output_model", type=Path)
    args = parser.parse_args()

    model = onnx.load(args.input_model)
    flattened = flatten_nested_concat(model)
    if not flattened:
        raise SystemExit("no exclusive same-axis nested Concat")
    model.producer_name = "ngc_flatten_nested_concat"
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output_model)
    print(f"flattened={flattened} output={args.output_model}")


if __name__ == "__main__":
    main()
