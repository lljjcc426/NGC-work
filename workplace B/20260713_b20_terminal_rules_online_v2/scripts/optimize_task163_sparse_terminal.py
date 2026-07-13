from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def dense_to_sparse(initializer: onnx.TensorProto) -> onnx.SparseTensorProto:
    array = numpy_helper.to_array(initializer)
    flat = array.reshape(-1)
    indices = np.flatnonzero(flat).astype(np.int64)
    values = flat[indices]
    if not len(indices):
        raise ValueError(f"initializer {initializer.name!r} is entirely zero")

    value_tensor = numpy_helper.from_array(values, name=initializer.name)
    index_tensor = numpy_helper.from_array(
        indices, name=f"{initializer.name}_indices"
    )
    return onnx.helper.make_sparse_tensor(
        value_tensor,
        index_tensor,
        list(array.shape),
    )


def optimize(source: Path, output: Path) -> None:
    model = onnx.load(source)
    if len(model.graph.node) != 1 or model.graph.node[0].op_type != "Einsum":
        raise ValueError("expected the single-terminal-Einsum task163 model")
    if model.graph.sparse_initializer:
        raise ValueError("source already contains sparse initializers")

    sparse = [dense_to_sparse(item) for item in model.graph.initializer]
    dense_params = sum(max(1, int(np.prod(item.dims))) for item in model.graph.initializer)
    sparse_params = sum(
        max(1, int(np.prod(item.values.dims))) for item in sparse
    )

    del model.graph.initializer[:]
    model.graph.sparse_initializer.extend(sparse)
    onnx.checker.check_model(model, full_check=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)
    print(
        f"dense_params={dense_params} sparse_params={sparse_params} "
        f"saved={output} bytes={output.stat().st_size}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    optimize(args.source, args.output)


if __name__ == "__main__":
    main()
