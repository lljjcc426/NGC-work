from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


B_TASKS = (
    1, 8, 18, 19, 23, 24, 56, 57, 63, 68, 76, 83, 88, 90, 97, 101,
    104, 123, 125, 128, 131, 134, 140, 143, 151, 161, 163, 170, 172,
    175, 181, 185, 205, 208, 209, 212, 228, 242, 244, 245, 247, 255,
    261, 266, 270, 277, 280, 285, 289, 291, 293, 295, 300, 308, 312,
    313, 317, 318, 320, 328, 344, 350, 360, 368, 369, 377, 395,
)


def exact_matrix_rank(array: np.ndarray) -> int:
    matrix = array.astype(np.float64, copy=False)
    return int(np.linalg.matrix_rank(matrix, tol=1e-9))


def rank_one_slices(array: np.ndarray) -> bool:
    if array.ndim != 3:
        return False
    return all(exact_matrix_rank(part) <= 1 for part in array)


def one_hot_factor_cost(array: np.ndarray) -> int | None:
    flat = array.reshape(array.shape[0], -1)
    if not np.all(np.count_nonzero(flat, axis=1) == 1):
        return None
    values = flat[np.arange(flat.shape[0]), np.argmax(flat != 0, axis=1)]
    if not np.all(values == 1):
        return None
    return array.shape[0] * sum(array.shape[1:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_dir",
        nargs="?",
        type=Path,
        default=Path(r"D:\golf\team_baselines\team_submission2_20260713\submission"),
    )
    args = parser.parse_args()

    for task in B_TASKS:
        path = args.model_dir / f"task{task:03d}.onnx"
        model = onnx.load(path)
        initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
        consumers: dict[str, list[str]] = defaultdict(list)
        equations: dict[str, list[str]] = defaultdict(list)
        for node in model.graph.node:
            for name in node.input:
                consumers[name].append(node.op_type)
                if node.op_type == "Einsum":
                    equation = next(
                        (onnx.helper.get_attribute_value(attr) for attr in node.attribute if attr.name == "equation"),
                        b"",
                    )
                    equations[name].append(equation.decode() if isinstance(equation, bytes) else str(equation))

        rows = []
        for name, array in initializers.items():
            if array.size < 24 or "Einsum" not in consumers[name] or array.ndim < 2:
                continue
            flat_rank = exact_matrix_rank(array.reshape(array.shape[0], -1))
            unique_slices = np.unique(array.reshape(array.shape[0], -1), axis=0).shape[0]
            hot_cost = one_hot_factor_cost(array)
            rows.append(
                (
                    -array.size,
                    name,
                    tuple(array.shape),
                    str(array.dtype),
                    array.size,
                    flat_rank,
                    unique_slices,
                    rank_one_slices(array),
                    hot_cost,
                    equations[name],
                )
            )
        if rows:
            print(f"task{task:03d}")
            for row in sorted(rows):
                _, name, shape, dtype, size, rank, unique, rank1, hot_cost, equations = row
                saving = size - hot_cost if hot_cost is not None else None
                print(
                    f"  {name}: shape={shape} dtype={dtype} size={size} "
                    f"axis0_rank={rank} unique0={unique} rank1_slices={rank1} "
                    f"onehot_factor_cost={hot_cost} saving={saving} equations={equations}"
                )


if __name__ == "__main__":
    main()
