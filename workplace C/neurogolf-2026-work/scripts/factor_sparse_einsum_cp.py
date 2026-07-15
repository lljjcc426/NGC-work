from __future__ import annotations

import argparse
import json
import string
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def get_equation(node: onnx.NodeProto) -> str | None:
    for attribute in node.attribute:
        if attribute.name == "equation":
            return attribute.s.decode("ascii")
    return None


def set_equation(node: onnx.NodeProto, equation: str) -> None:
    for attribute in node.attribute:
        if attribute.name == "equation":
            attribute.s = equation.encode("ascii")
            return
    raise RuntimeError("Einsum equation attribute missing")


def _factor_first_axis(array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Factor a rank-3 tensor by exact rank decompositions of its first-axis slices.

    This deliberately accepts only integer-valued float tensors. Gaussian elimination
    is then exact in float64 for the small integer weights used by NeuroGolf models,
    and the reconstructed float32 tensor must match bit-for-bit before promotion.
    """
    if array.ndim != 3 or array.dtype not in (np.float16, np.float32, np.float64):
        return None
    source = array.astype(np.float64)
    if not np.array_equal(source, np.rint(source)):
        return None

    first_factors: list[np.ndarray] = []
    second_factors: list[np.ndarray] = []
    third_factors: list[np.ndarray] = []
    for slice_index, matrix in enumerate(source):
        if not np.any(matrix):
            continue
        # SVD is used only to locate the numerical rank. The actual factors are
        # selected source rows and exact least-squares coefficients, followed by
        # strict integer reconstruction checks below.
        rank = int(np.linalg.matrix_rank(matrix))
        selected: list[int] = []
        for row_index in range(matrix.shape[0]):
            trial = matrix[selected + [row_index]]
            if np.linalg.matrix_rank(trial) > len(selected):
                selected.append(row_index)
            if len(selected) == rank:
                break
        if len(selected) != rank:
            return None
        row_basis = matrix[selected]
        coefficients = np.linalg.lstsq(row_basis.T, matrix.T, rcond=None)[0].T
        coefficients = np.rint(coefficients)
        if not np.array_equal(coefficients @ row_basis, matrix):
            return None
        for component in range(rank):
            axis0 = np.zeros(array.shape[0], dtype=np.float32)
            axis0[slice_index] = 1.0
            first_factors.append(axis0)
            second_factors.append(coefficients[:, component].astype(np.float32))
            third_factors.append(row_basis[component].astype(np.float32))

    if not first_factors:
        return None
    first = np.stack(first_factors, axis=1).astype(array.dtype)
    second = np.stack(second_factors, axis=1).astype(array.dtype)
    third = np.stack(third_factors, axis=1).astype(array.dtype)
    reconstructed = np.einsum("ar,br,cr->abc", first, second, third)
    if not np.array_equal(reconstructed, array):
        return None
    if first.size + second.size + third.size >= array.size:
        return None
    return first, second, third


def exact_slice_cp(array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    best: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    for first_axis in range(3):
        order = [first_axis] + [axis for axis in range(3) if axis != first_axis]
        moved = np.transpose(array, order)
        moved_factors = _factor_first_axis(moved)
        if moved_factors is None:
            continue
        factors_by_original_axis: list[np.ndarray | None] = [None, None, None]
        for moved_axis, original_axis in enumerate(order):
            factors_by_original_axis[original_axis] = moved_factors[moved_axis]
        factors = tuple(factors_by_original_axis)
        if any(factor is None for factor in factors):
            continue
        typed = (factors[0], factors[1], factors[2])
        reconstructed = np.einsum("ar,br,cr->abc", *typed)
        if not np.array_equal(reconstructed, array):
            continue
        if best is None or sum(factor.size for factor in typed) < sum(factor.size for factor in best):
            best = typed
    return best


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
        for character in (get_equation(node) or "")
        if character.isalpha()
    }
    labels = [character for character in string.ascii_letters if character not in used_labels]
    remove_names: set[str] = set()
    add_tensors: list[onnx.TensorProto] = []
    changes: list[dict] = []

    for node_index, node in enumerate(result.graph.node):
        if node.op_type != "Einsum":
            continue
        raw_equation = get_equation(node)
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
                or len(tensor.dims) != 3
                or len(subscripts) != 3
                or len(set(subscripts)) != 3
                or not labels
            ):
                input_index += 1
                continue
            array = numpy_helper.to_array(tensor)
            factors = exact_slice_cp(array)
            if factors is None:
                input_index += 1
                continue
            first, second, third = factors
            rank_label = labels.pop(0)
            factor_names = [f"{name}_cp_axis{axis}" for axis in range(3)]
            used_names = set(initializers)
            suffix = 0
            while any(factor_name in used_names for factor_name in factor_names):
                suffix += 1
                factor_names = [f"{name}_cp_axis{axis}_{suffix}" for axis in range(3)]
            factor_subscripts = [subscripts[axis] + rank_label for axis in range(3)]
            operands[input_index : input_index + 1] = factor_subscripts
            inputs = list(node.input)
            inputs[input_index : input_index + 1] = factor_names
            del node.input[:]
            node.input.extend(inputs)
            remove_names.add(name)
            add_tensors.extend(
                numpy_helper.from_array(factor, factor_name)
                for factor, factor_name in zip(factors, factor_names)
            )
            changes.append(
                {
                    "node_index": node_index,
                    "initializer": name,
                    "old_shape": list(array.shape),
                    "factor_shapes": [list(factor.shape) for factor in factors],
                    "rank": int(first.shape[1]),
                    "removed_parameters": int(array.size - sum(factor.size for factor in factors)),
                }
            )
            input_index += 3
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
