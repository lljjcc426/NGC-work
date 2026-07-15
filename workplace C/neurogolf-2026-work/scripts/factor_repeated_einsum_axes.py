from __future__ import annotations

import argparse
import json
import string
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def equation(node: onnx.NodeProto) -> str | None:
    for attribute in node.attribute:
        if attribute.name == "equation":
            return attribute.s.decode("ascii")
    return None


def set_equation(node: onnx.NodeProto, value: str) -> None:
    for attribute in node.attribute:
        if attribute.name == "equation":
            attribute.s = value.encode("ascii")
            return
    raise RuntimeError("Einsum equation attribute missing")


def unique_axis_factor(array: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray] | None:
    moved = np.moveaxis(array, axis, 0)
    flat = moved.reshape(moved.shape[0], -1)
    unique, inverse = np.unique(flat, axis=0, return_inverse=True)
    if unique.shape[0] == moved.shape[0]:
        return None
    core_moved = unique.reshape((unique.shape[0],) + moved.shape[1:])
    core = np.moveaxis(core_moved, 0, axis)
    selector = np.zeros((moved.shape[0], unique.shape[0]), dtype=array.dtype)
    selector[np.arange(moved.shape[0]), inverse] = 1
    return selector, core


def transform(model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[dict]]:
    result = onnx.ModelProto()
    result.CopyFrom(model)
    initializers = {tensor.name: tensor for tensor in result.graph.initializer}
    consumers: dict[str, int] = {}
    for node in result.graph.node:
        for name in node.input:
            consumers[name] = consumers.get(name, 0) + 1
    changes = []
    remove_names = set()
    add_tensors = []
    used_names = {value.name for value in result.graph.input}
    used_names.update(value.name for value in result.graph.output)
    used_names.update(initializers)
    used_labels = set()
    for node in result.graph.node:
        if node.op_type == "Einsum" and equation(node):
            used_labels.update(ch for ch in equation(node) or "" if ch.isalpha())

    available_labels = [ch for ch in string.ascii_letters if ch not in used_labels]
    for node_index, node in enumerate(result.graph.node):
        if node.op_type != "Einsum":
            continue
        raw_equation = equation(node)
        if not raw_equation or "->" not in raw_equation:
            continue
        left, output = raw_equation.split("->", 1)
        operands = left.split(",")
        if len(operands) != len(node.input):
            continue
        input_index = 0
        while input_index < len(node.input):
            name = node.input[input_index]
            tensor = initializers.get(name)
            subscripts = operands[input_index]
            if (
                tensor is None
                or consumers.get(name) != 1
                or "..." in subscripts
                or len(subscripts) != len(tensor.dims)
                or len(set(subscripts)) != len(subscripts)
            ):
                input_index += 1
                continue
            array = numpy_helper.to_array(tensor)
            best = None
            for axis in range(array.ndim):
                factor = unique_axis_factor(array, axis)
                if factor is None:
                    continue
                selector, core = factor
                new_params = selector.size + core.size
                gain = array.size - new_params
                if gain > 0 and (best is None or gain > best[0]):
                    best = (gain, axis, selector, core)
            if best is None or not available_labels:
                input_index += 1
                continue
            gain, axis, selector, core = best
            label = available_labels.pop(0)
            selector_name = f"{name}_axis{axis}_selector"
            core_name = f"{name}_axis{axis}_core"
            suffix = 0
            while selector_name in used_names or core_name in used_names:
                suffix += 1
                selector_name = f"{name}_axis{axis}_selector_{suffix}"
                core_name = f"{name}_axis{axis}_core_{suffix}"
            used_names.update((selector_name, core_name))
            core_subscripts = subscripts[:axis] + label + subscripts[axis + 1 :]
            selector_subscripts = subscripts[axis] + label
            operands[input_index : input_index + 1] = [selector_subscripts, core_subscripts]
            inputs = list(node.input)
            inputs[input_index : input_index + 1] = [selector_name, core_name]
            del node.input[:]
            node.input.extend(inputs)
            remove_names.add(name)
            add_tensors.extend(
                [
                    numpy_helper.from_array(selector, selector_name),
                    numpy_helper.from_array(core, core_name),
                ]
            )
            changes.append(
                {
                    "node_index": node_index,
                    "initializer": name,
                    "axis": axis,
                    "old_shape": list(array.shape),
                    "selector_shape": list(selector.shape),
                    "core_shape": list(core.shape),
                    "removed_parameters": int(gain),
                }
            )
            input_index += 2
        set_equation(node, ",".join(operands) + "->" + output)

    if changes:
        kept = [tensor for tensor in result.graph.initializer if tensor.name not in remove_names]
        del result.graph.initializer[:]
        result.graph.initializer.extend(kept + add_tensors)
    return result, changes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = (
        [args.parent_dir / f"{args.task}.onnx"]
        if args.task
        else sorted(args.parent_dir.glob("task*.onnx"))
    )
    rows = []
    for path in paths:
        model = onnx.load(path)
        candidate, changes = transform(model)
        if not changes:
            continue
        output = args.output_dir / path.name
        try:
            onnx.checker.check_model(candidate, full_check=True)
            onnx.shape_inference.infer_shapes(candidate, strict_mode=True)
            onnx.save(candidate, output)
            rows.append(
                {
                    "task": path.stem,
                    "output": str(output),
                    "removed_parameters": sum(c["removed_parameters"] for c in changes),
                    "changes": changes,
                }
            )
        except Exception as exc:
            rows.append({"task": path.stem, "error": f"{type(exc).__name__}:{exc}", "changes": changes})
    rows.sort(key=lambda row: -int(row.get("removed_parameters", 0)))
    print(json.dumps({"changed": len([r for r in rows if "output" in r]), "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
