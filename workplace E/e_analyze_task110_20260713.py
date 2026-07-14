from __future__ import annotations

import argparse
import csv
import hashlib
import math
from collections import Counter
from pathlib import Path

import numpy as np
import onnx


FIELDS = [
    "sha256",
    "path_count",
    "representative",
    "filesize",
    "nodes",
    "ops",
    "memory_static",
    "params",
    "cost_static",
    "error",
]


def tensor_elements(value: onnx.ValueInfoProto) -> int:
    tensor_type = value.type.tensor_type
    if not tensor_type.HasField("shape"):
        raise ValueError(f"missing shape for {value.name}")
    dims = [dim.dim_value for dim in tensor_type.shape.dim]
    if any(dim <= 0 for dim in dims):
        raise ValueError(f"dynamic shape for {value.name}: {dims}")
    return math.prod(dims)


def static_memory(model: onnx.ModelProto) -> int:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    graph = inferred.graph
    values = {
        value.name: value
        for value in [*graph.input, *graph.value_info, *graph.output]
    }
    total = 0
    for node in graph.node:
        for output in node.output:
            if not output or output == "output":
                continue
            value = values[output]
            dtype = onnx.helper.tensor_dtype_to_np_dtype(
                value.type.tensor_type.elem_type
            )
            total += tensor_elements(value) * np.dtype(dtype).itemsize
    return total


def parameter_count(model: onnx.ModelProto) -> int:
    total = sum(math.prod(init.dims) if init.dims else 1 for init in model.graph.initializer)
    total += sum(
        math.prod(init.values.dims) if init.values.dims else 1
        for init in model.graph.sparse_initializer
    )
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attribute in node.attribute:
            if attribute.name == "value":
                total += math.prod(attribute.t.dims) if attribute.t.dims else 1
            elif attribute.name == "sparse_value":
                dims = attribute.sparse_tensor.values.dims
                total += math.prod(dims) if dims else 1
            elif attribute.name == "value_floats":
                total += len(attribute.floats)
            elif attribute.name == "value_ints":
                total += len(attribute.ints)
            elif attribute.name == "value_strings":
                total += len(attribute.strings)
    return total


def summarize(paths: list[Path]) -> dict[str, object]:
    representative = paths[0]
    row: dict[str, object] = {
        "sha256": hashlib.sha256(representative.read_bytes()).hexdigest(),
        "path_count": len(paths),
        "representative": str(representative),
        "filesize": representative.stat().st_size,
        "nodes": "",
        "ops": "",
        "memory_static": "",
        "params": "",
        "cost_static": "",
        "error": "",
    }
    try:
        model = onnx.load(representative)
        memory = static_memory(model)
        params = parameter_count(model)
        ops = Counter(node.op_type for node in model.graph.node)
        row.update(
            {
                "nodes": len(model.graph.node),
                "ops": ";".join(f"{name}:{count}" for name, count in sorted(ops.items())),
                "memory_static": memory,
                "params": params,
                "cost_static": memory + params,
            }
        )
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    grouped: dict[str, list[Path]] = {}
    for root in args.roots:
        for path in root.rglob("task110.onnx"):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            grouped.setdefault(digest, []).append(path)

    rows = [summarize(sorted(paths)) for paths in grouped.values()]
    rows.sort(
        key=lambda row: (
            row["cost_static"] == "",
            int(row["cost_static"]) if row["cost_static"] != "" else 10**18,
            str(row["representative"]),
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    valid = sum(not row["error"] for row in rows)
    print(f"models={sum(len(paths) for paths in grouped.values())}")
    print(f"unique={len(rows)} valid={valid}")
    for row in rows[:20]:
        print(
            row["cost_static"],
            row["nodes"],
            row["path_count"],
            row["representative"],
        )


if __name__ == "__main__":
    main()
