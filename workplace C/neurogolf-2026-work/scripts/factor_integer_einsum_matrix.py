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


def exact_matrix_factor(array: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if array.ndim != 2 or array.dtype not in (np.float16, np.float32, np.float64):
        return None
    source = array.astype(np.float64)
    if not np.array_equal(source, np.rint(source)):
        return None
    rank = int(np.linalg.matrix_rank(source))
    if rank == 0 or rank * sum(source.shape) >= source.size:
        return None
    selected: list[int] = []
    for row_index in range(source.shape[0]):
        trial = source[selected + [row_index]]
        if np.linalg.matrix_rank(trial) > len(selected):
            selected.append(row_index)
        if len(selected) == rank:
            break
    if len(selected) != rank:
        return None
    right = source[selected]
    left = np.rint(np.linalg.lstsq(right.T, source.T, rcond=None)[0].T)
    if not np.array_equal(left @ right, source):
        return None
    left = left.astype(array.dtype)
    right = right.astype(array.dtype)
    if not np.array_equal(left @ right, array):
        return None
    return left, right


def transform(model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[dict]]:
    result = onnx.ModelProto()
    result.CopyFrom(model)
    initializers = {tensor.name: tensor for tensor in result.graph.initializer}
    consumers: dict[str, int] = {}
    for node in result.graph.node:
        for name in node.input:
            consumers[name] = consumers.get(name, 0) + 1
    used_labels = {
        character
        for node in result.graph.node
        if node.op_type == "Einsum"
        for character in (equation(node) or "")
        if character.isalpha()
    }
    labels = [character for character in string.ascii_letters if character not in used_labels]
    remove_names: set[str] = set()
    add_tensors: list[onnx.TensorProto] = []
    changes: list[dict] = []
    for node_index, node in enumerate(result.graph.node):
        if node.op_type != "Einsum":
            continue
        raw = equation(node)
        if not raw or "->" not in raw:
            continue
        left_equation, output = raw.split("->", 1)
        operands = left_equation.split(",")
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
                or len(tensor.dims) != 2
                or len(subscripts) != 2
                or len(set(subscripts)) != 2
                or not labels
            ):
                input_index += 1
                continue
            array = numpy_helper.to_array(tensor)
            factors = exact_matrix_factor(array)
            if factors is None:
                input_index += 1
                continue
            left, right = factors
            rank_label = labels.pop(0)
            names = [f"{name}_rank_left", f"{name}_rank_right"]
            operands[input_index : input_index + 1] = [subscripts[0] + rank_label, rank_label + subscripts[1]]
            inputs = list(node.input)
            inputs[input_index : input_index + 1] = names
            del node.input[:]
            node.input.extend(inputs)
            remove_names.add(name)
            add_tensors.extend(
                [numpy_helper.from_array(left, names[0]), numpy_helper.from_array(right, names[1])]
            )
            changes.append(
                {
                    "node_index": node_index,
                    "initializer": name,
                    "old_shape": list(array.shape),
                    "factor_shapes": [list(left.shape), list(right.shape)],
                    "rank": int(left.shape[1]),
                    "removed_parameters": int(array.size - left.size - right.size),
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
    paths = [args.parent_dir / f"{args.task}.onnx"] if args.task else sorted(args.parent_dir.glob("task*.onnx"))
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
                    "removed_parameters": sum(change["removed_parameters"] for change in changes),
                    "changes": changes,
                }
            )
        except Exception as exc:
            rows.append({"task": path.stem, "error": f"{type(exc).__name__}:{exc}", "changes": changes})
    rows.sort(key=lambda row: -int(row.get("removed_parameters", 0)))
    print(json.dumps({"changed": sum("output" in row for row in rows), "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
