from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto, numpy_helper


HERE = Path(__file__).resolve()


DTYPE_BYTES = {
    TensorProto.BOOL: 1,
    TensorProto.FLOAT16: 2,
    TensorProto.BFLOAT16: 2,
    TensorProto.INT8: 1,
    TensorProto.UINT8: 1,
    TensorProto.INT16: 2,
    TensorProto.UINT16: 2,
    TensorProto.FLOAT: 4,
    TensorProto.INT32: 4,
    TensorProto.UINT32: 4,
    TensorProto.DOUBLE: 8,
    TensorProto.INT64: 8,
    TensorProto.UINT64: 8,
}


def _tensor_metadata(model: onnx.ModelProto) -> dict[str, dict[str, Any]]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    result: dict[str, dict[str, Any]] = {}
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    for value in values:
        tensor = value.type.tensor_type
        shape: list[int] = []
        known = True
        for dim in tensor.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(int(dim.dim_value))
            else:
                known = False
                shape.append(-1)
        elements = 1
        if not known:
            elements = -1
        else:
            for dim in shape:
                elements *= dim
        item_size = DTYPE_BYTES.get(tensor.elem_type)
        result[value.name] = {
            "shape": shape,
            "dtype": TensorProto.DataType.Name(tensor.elem_type).lower(),
            "elements": elements,
            "bytes": elements * item_size if elements >= 0 and item_size is not None else None,
        }
    for initializer in inferred.graph.initializer:
        array = numpy_helper.to_array(initializer)
        result[initializer.name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "elements": int(array.size),
            "bytes": int(array.nbytes),
            "initializer": True,
        }
    return result


def profile_model(path: Path) -> dict[str, Any]:
    model = onnx.load(path)
    metadata = _tensor_metadata(model)
    initializer_names = {item.name for item in model.graph.initializer}
    graph_outputs = {item.name for item in model.graph.output}
    consumers = Counter(name for node in model.graph.node for name in node.input if name)
    producer = {
        name: index
        for index, node in enumerate(model.graph.node)
        for name in node.output
        if name
    }
    live = {
        item.name
        for item in model.graph.input
        if item.name not in initializer_names
    }
    rows: list[dict[str, Any]] = []
    peak_bytes = 0
    peak_node = -1
    unknown_live: set[str] = set()

    def live_size(names: set[str]) -> tuple[int, list[str]]:
        total = 0
        unknown: list[str] = []
        for name in names:
            size = metadata.get(name, {}).get("bytes")
            if size is None:
                unknown.append(name)
            else:
                total += int(size)
        return total, sorted(unknown)

    for index, node in enumerate(model.graph.node):
        before = set(live)
        outputs = {name for name in node.output if name and name not in initializer_names}
        during = before | outputs
        during_bytes, unknown = live_size(during)
        if during_bytes > peak_bytes:
            peak_bytes = during_bytes
            peak_node = index
            unknown_live = set(unknown)
        released: list[str] = []
        for name in node.input:
            if not name or name in initializer_names:
                continue
            consumers[name] -= 1
            if consumers[name] <= 0 and name not in graph_outputs:
                live.discard(name)
                released.append(name)
        live.update(outputs)
        after_bytes, after_unknown = live_size(live)
        rows.append({
            "node_index": index,
            "node_name": node.name,
            "op_type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "live_before": sorted(before),
            "released_after": sorted(released),
            "live_after": sorted(live),
            "during_bytes": during_bytes,
            "after_bytes": after_bytes,
            "unknown_during": unknown,
            "unknown_after": after_unknown,
        })

    largest_tensors = sorted(
        (
            {"name": name, **info, "producer_node": producer.get(name)}
            for name, info in metadata.items()
            if name not in initializer_names and info.get("bytes") is not None
        ),
        key=lambda item: (-int(item["bytes"]), item["name"]),
    )
    largest_initializers = sorted(
        (
            {"name": name, **info}
            for name, info in metadata.items()
            if name in initializer_names
        ),
        key=lambda item: (-int(item.get("bytes", 0)), item["name"]),
    )
    return {
        "model": str(path.resolve()),
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "initializer_bytes": sum(item.ByteSize() for item in model.graph.initializer),
        "activation_peak_bytes": peak_bytes,
        "peak_node_index": peak_node,
        "peak_node": rows[peak_node] if peak_node >= 0 else None,
        "peak_has_unknown_tensor": bool(unknown_live),
        "largest_tensors": largest_tensors[:10],
        "largest_initializers": largest_initializers[:10],
        "nodes": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate ONNX activation peak from tensor liveness.")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--parent-model", type=Path)
    parser.add_argument("--candidate-model", type=Path)
    parser.add_argument("--onnx-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths: list[Path]
    if args.parent_model and args.candidate_model:
        paths = [args.parent_model, args.candidate_model]
    elif args.model:
        paths = [args.model]
    elif args.onnx_dir:
        paths = sorted(args.onnx_dir.glob("task*.onnx"))
    else:
        parser.error("provide --model, --parent-model/--candidate-model, or --onnx-dir")
    profiles = [profile_model(path) for path in paths]
    payload: dict[str, Any] = {"profiles": profiles}
    if len(profiles) == 2:
        payload["peak_delta_bytes"] = (
            profiles[1]["activation_peak_bytes"] - profiles[0]["activation_peak_bytes"]
        )
    if args.output:
        from full400_safety import atomic_write_json

        atomic_write_json(args.output, payload)
    summary = {
        "models": len(profiles),
        "profiles": [
            {
                "model": item["model"],
                "node_count": item["node_count"],
                "activation_peak_bytes": item["activation_peak_bytes"],
                "peak_node_index": item["peak_node_index"],
                "peak_op_type": item["peak_node"]["op_type"] if item["peak_node"] else None,
            }
            for item in profiles
        ],
    }
    if "peak_delta_bytes" in payload:
        summary["peak_delta_bytes"] = payload["peak_delta_bytes"]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
