from __future__ import annotations

import argparse
from itertools import permutations
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def _equation(node: onnx.NodeProto) -> tuple[onnx.AttributeProto, list[str], str]:
    attr = next(item for item in node.attribute if item.name == "equation")
    lhs, rhs = attr.s.decode("ascii").split("->", 1)
    return attr, lhs.split(","), rhs


def _source_term(target_term: str, permutation: tuple[int, ...]) -> str | None:
    if "." in target_term or len(target_term) != len(permutation):
        return None
    source = [""] * len(permutation)
    for target_axis, source_axis in enumerate(permutation):
        source[source_axis] = target_term[target_axis]
    return "".join(source)


def build(source_path: Path, output_path: Path) -> tuple[Path, list[dict[str, object]]]:
    model = onnx.load(source_path)
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    initializer_order = [item.name for item in model.graph.initializer]
    consumers: dict[str, list[tuple[onnx.NodeProto, int]]] = {name: [] for name in arrays}
    for node in model.graph.node:
        for input_index, name in enumerate(node.input):
            if name in consumers:
                consumers[name].append((node, input_index))

    removed: set[str] = set()
    changes: list[dict[str, object]] = []
    for target_index, target_name in enumerate(initializer_order):
        target = arrays[target_name]
        uses = consumers[target_name]
        if target_name in removed or target.ndim < 2 or target.ndim > 4 or not uses:
            continue
        if any(node.op_type != "Einsum" for node, _ in uses):
            continue
        match: tuple[str, tuple[int, ...]] | None = None
        for source_name in initializer_order[:target_index]:
            if source_name in removed:
                continue
            source = arrays[source_name]
            if source.ndim != target.ndim or source.size != target.size:
                continue
            for permutation in permutations(range(source.ndim)):
                if permutation == tuple(range(source.ndim)):
                    continue
                if tuple(source.shape[axis] for axis in permutation) != target.shape:
                    continue
                if np.array_equal(np.transpose(source, permutation), target):
                    match = source_name, permutation
                    break
            if match:
                break
        if match is None:
            continue

        source_name, permutation = match
        rewritten: list[tuple[onnx.NodeProto, int, str]] = []
        valid = True
        for node, input_index in uses:
            _, terms, _ = _equation(node)
            source_term = _source_term(terms[input_index], permutation)
            if source_term is None:
                valid = False
                break
            rewritten.append((node, input_index, source_term))
        if not valid:
            continue
        # Apply all substitutions for one Einsum together. Re-parsing and
        # writing the equation per occurrence loses earlier rewrites when the
        # same initializer is referenced multiple times by the same node.
        by_node: dict[int, tuple[onnx.NodeProto, list[tuple[int, str]]]] = {}
        for node, input_index, source_term in rewritten:
            key = id(node)
            by_node.setdefault(key, (node, []))[1].append((input_index, source_term))
        for node, substitutions in by_node.values():
            attr, terms, rhs = _equation(node)
            for input_index, source_term in substitutions:
                node.input[input_index] = source_name
                terms[input_index] = source_term
            attr.s = (",".join(terms) + "->" + rhs).encode("ascii")
        removed.add(target_name)
        changes.append(
            {
                "removed": target_name,
                "source": source_name,
                "permutation": list(permutation),
                "saved_parameters": int(target.size),
            }
        )

    if not changes:
        raise RuntimeError("no transposed Einsum initializer duplicates found")
    kept = [item for item in model.graph.initializer if item.name not in removed]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(inferred, output_path)
    reloaded = onnx.load(output_path)
    onnx.checker.check_model(reloaded, full_check=True)
    onnx.shape_inference.infer_shapes(reloaded, strict_mode=True, data_prop=True)
    return output_path, changes


def main() -> int:
    parser = argparse.ArgumentParser(description="Deduplicate exact transposed initializers used only by Einsum nodes.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output, changes = build(args.source, args.output)
    print({"output": str(output), "changes": changes})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
