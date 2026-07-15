from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def _equation(node: onnx.NodeProto) -> onnx.AttributeProto:
    return next(attr for attr in node.attribute if attr.name == "equation")


def build(source: Path, output: Path) -> tuple[Path, list[dict[str, object]]]:
    model = onnx.load(source)
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    use_counts = Counter(value for node in model.graph.node for value in node.input)
    removed: set[str] = set()
    additions: list[onnx.TensorProto] = []
    changes: list[dict[str, object]] = []

    for node_index, node in enumerate(model.graph.node):
        if node.op_type != "Einsum":
            continue
        attribute = _equation(node)
        lhs, rhs = attribute.s.decode("ascii").split("->", 1)
        terms = lhs.split(",")
        if len(terms) != len(node.input) or any("." in term for term in terms):
            continue
        constant_indices = [
            index
            for index, name in enumerate(node.input)
            if name in arrays and use_counts[name] == 1
        ]
        external_indices = set(range(len(terms))) - set(constant_indices)
        label_owners: dict[str, list[int]] = defaultdict(list)
        for index, term in enumerate(terms):
            for label in set(term):
                label_owners[label].append(index)

        adjacency = {index: set() for index in constant_indices}
        for label, owners in label_owners.items():
            if label in rhs or any(index in external_indices for index in owners):
                continue
            constant_owners = [index for index in owners if index in adjacency]
            for index in constant_owners:
                adjacency[index].update(other for other in constant_owners if other != index)

        components: list[list[int]] = []
        seen: set[int] = set()
        for start in constant_indices:
            if start in seen:
                continue
            stack = [start]
            component: list[int] = []
            while stack:
                index = stack.pop()
                if index in seen:
                    continue
                seen.add(index)
                component.append(index)
                stack.extend(adjacency[index] - seen)
            if len(component) >= 2:
                components.append(sorted(component))

        replacements: list[tuple[list[int], str, str, np.ndarray, int]] = []
        for component_index, component in enumerate(components):
            outside_labels = set(rhs)
            for index, term in enumerate(terms):
                if index not in component:
                    outside_labels.update(term)
            output_labels: list[str] = []
            for index in component:
                for label in terms[index]:
                    if label in outside_labels and label not in output_labels:
                        output_labels.append(label)
            component_equation = (
                ",".join(terms[index] for index in component)
                + "->"
                + "".join(output_labels)
            )
            result = np.einsum(
                component_equation,
                *(arrays[node.input[index]] for index in component),
                optimize=True,
            )
            old_parameters = sum(arrays[node.input[index]].size for index in component)
            if result.size >= old_parameters:
                continue
            name = f"constant_component_{node_index}_{component_index}"
            replacements.append((component, name, "".join(output_labels), result, old_parameters))

        inputs = list(node.input)
        for component, name, output_term, result, old_parameters in sorted(
            replacements, key=lambda item: min(item[0]), reverse=True
        ):
            first = min(component)
            names = [inputs[index] for index in component]
            inputs[first] = name
            terms[first] = output_term
            for index in sorted((index for index in component if index != first), reverse=True):
                del inputs[index]
                del terms[index]
            removed.update(names)
            additions.append(numpy_helper.from_array(result, name))
            changes.append(
                {
                    "node_index": node_index,
                    "removed_initializers": names,
                    "new_initializer": name,
                    "old_parameters": int(old_parameters),
                    "new_parameters": int(result.size),
                    "saved_parameters": int(old_parameters - result.size),
                }
            )
        if replacements:
            del node.input[:]
            node.input.extend(inputs)
            attribute.s = (",".join(terms) + "->" + rhs).encode("ascii")

    if not changes:
        raise RuntimeError("no profitable terminal constant component found")
    kept = [item for item in model.graph.initializer if item.name not in removed]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.initializer.extend(additions)
    # Node outputs and their shapes are unchanged. Preserve source value_info:
    # some legacy graphs contain static dimensions that standard ONNX shape
    # inference cannot reconstruct, and the official memory scorer requires
    # those concrete shapes.
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(inferred, output)
    reloaded = onnx.load(output)
    onnx.checker.check_model(reloaded, full_check=True)
    onnx.shape_inference.infer_shapes(reloaded, strict_mode=True, data_prop=True)
    return output, changes


def main() -> int:
    parser = argparse.ArgumentParser(description="Precontract profitable constant-only components inside terminal Einsum nodes.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    path, changes = build(args.source, args.output)
    print({"output": str(path), "changes": changes})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
