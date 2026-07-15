from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve()
PROJECT = HERE.parent.parent
DEFAULT_OUTPUT = PROJECT / "config" / "runtime_capability_matrix.json"
DEFAULT_PROBE_DIR = PROJECT.parent / "artifacts" / "runtime_capability_probes"


DTYPES = {
    "float16": (TensorProto.FLOAT16, np.float16),
    "float32": (TensorProto.FLOAT, np.float32),
    "int8": (TensorProto.INT8, np.int8),
    "int16": (TensorProto.INT16, np.int16),
    "int32": (TensorProto.INT32, np.int32),
    "int64": (TensorProto.INT64, np.int64),
    "uint8": (TensorProto.UINT8, np.uint8),
}


PROBES = (
    {"name": "TopK:float16", "operator": "TopK", "dtype": "float16"},
    {"name": "TopK:int16", "operator": "TopK", "dtype": "int16"},
    {"name": "TopK:int32", "operator": "TopK", "dtype": "int32"},
    {"name": "TopK:int64", "operator": "TopK", "dtype": "int64"},
    {"name": "TopK:int8", "operator": "TopK", "dtype": "int8", "kaggle_status": "ERROR", "kaggle_ref": 54660619},
    {"name": "TopK:uint8", "operator": "TopK", "dtype": "uint8", "kaggle_status": "ERROR"},
    {"name": "Einsum:float16", "operator": "Einsum", "dtype": "float16"},
    {"name": "Einsum:uint8", "operator": "Einsum", "dtype": "uint8"},
    {"name": "Where:int8", "operator": "Where", "dtype": "int8"},
    {"name": "Where:int16", "operator": "Where", "dtype": "int16"},
    {"name": "Where:int64", "operator": "Where", "dtype": "int64"},
    {"name": "Sum:int32:variadic", "operator": "Sum", "dtype": "int32"},
    {"name": "Sum:float16:variadic", "operator": "Sum", "dtype": "float16"},
    {"name": "ScatterND:int32", "operator": "ScatterND", "dtype": "float32", "index_dtype": "int32"},
    {"name": "ScatterND:int64", "operator": "ScatterND", "dtype": "float32", "index_dtype": "int64"},
)


def _value(name: str, dtype: str, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, DTYPES[dtype][0], shape)


def build_probe(spec: dict[str, Any]) -> tuple[onnx.ModelProto, dict[str, np.ndarray]]:
    operator = spec["operator"]
    dtype = spec["dtype"]
    np_dtype = DTYPES[dtype][1]
    initializers: list[onnx.TensorProto] = []
    inputs: list[onnx.ValueInfoProto] = []
    outputs: list[onnx.ValueInfoProto] = []
    feeds: dict[str, np.ndarray] = {}

    if operator == "TopK":
        inputs = [_value("x", dtype, [1, 4])]
        outputs = [_value("values", dtype, [1, 2]), _value("indices", "int64", [1, 2])]
        initializers = [numpy_helper.from_array(np.asarray([2], dtype=np.int64), name="k")]
        nodes = [helper.make_node("TopK", ["x", "k"], ["values", "indices"], axis=1)]
        feeds = {"x": np.asarray([[1, 4, 2, 3]], dtype=np_dtype)}
    elif operator == "Einsum":
        inputs = [_value("x", dtype, [2, 2])]
        outputs = [_value("y", dtype, [2, 2])]
        nodes = [helper.make_node("Einsum", ["x"], ["y"], equation="ij->ji")]
        feeds = {"x": np.asarray([[1, 2], [3, 4]], dtype=np_dtype)}
    elif operator == "Where":
        inputs = [_value("condition", "uint8", [2]), _value("x", dtype, [2]), _value("y", dtype, [2])]
        inputs[0].type.tensor_type.elem_type = TensorProto.BOOL
        outputs = [_value("z", dtype, [2])]
        nodes = [helper.make_node("Where", ["condition", "x", "y"], ["z"])]
        feeds = {
            "condition": np.asarray([True, False], dtype=np.bool_),
            "x": np.asarray([1, 2], dtype=np_dtype),
            "y": np.asarray([3, 4], dtype=np_dtype),
        }
    elif operator == "Sum":
        inputs = [_value(name, dtype, [2]) for name in ("a", "b", "c")]
        outputs = [_value("z", dtype, [2])]
        nodes = [helper.make_node("Sum", ["a", "b", "c"], ["z"])]
        feeds = {name: np.asarray([index, index + 1], dtype=np_dtype) for index, name in enumerate(("a", "b", "c"), 1)}
    elif operator == "ScatterND":
        index_dtype = spec["index_dtype"]
        inputs = [_value("data", dtype, [4]), _value("indices", index_dtype, [2, 1]), _value("updates", dtype, [2])]
        outputs = [_value("output", dtype, [4])]
        nodes = [helper.make_node("ScatterND", ["data", "indices", "updates"], ["output"])]
        feeds = {
            "data": np.asarray([0, 0, 0, 0], dtype=np_dtype),
            "indices": np.asarray([[1], [3]], dtype=DTYPES[index_dtype][1]),
            "updates": np.asarray([5, 7], dtype=np_dtype),
        }
    else:
        raise ValueError(operator)

    graph = helper.make_graph(nodes, f"probe_{spec['name']}", inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    return model, feeds


def run_probe(spec: dict[str, Any], probe_dir: Path) -> dict[str, Any]:
    model, feeds = build_probe(spec)
    path = probe_dir / f"{spec['name'].replace(':', '_')}.onnx"
    row = dict(spec)
    row.update({
        "opset": 18,
        "local_checker": False,
        "strict_shape_inference": False,
        "official_sanitizer": False,
        "local_ort": False,
        "output_dtypes": [],
        "error": "",
    })
    errors: list[str] = []
    try:
        onnx.checker.check_model(model, full_check=True)
        row["local_checker"] = True
    except Exception as exc:
        errors.append(f"checker:{type(exc).__name__}:{exc}")
    try:
        model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
        row["strict_shape_inference"] = True
    except Exception as exc:
        errors.append(f"shape:{type(exc).__name__}:{exc}")
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    try:
        from c_score_common import load_official_utils

        sanitized = load_official_utils().sanitize_model(model)
        row["official_sanitizer"] = sanitized is not None
        if sanitized is not None:
            model = sanitized
    except Exception as exc:
        errors.append(f"sanitizer:{type(exc).__name__}:{exc}")
    try:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
        session_inputs = session.get_inputs()
        if len(session_inputs) != len(feeds):
            raise RuntimeError(
                f"sanitizer changed input count: {len(feeds)} -> {len(session_inputs)}"
            )
        runtime_feeds = {
            item.name: value
            for item, value in zip(session_inputs, feeds.values(), strict=True)
        }
        values = session.run(None, runtime_feeds)
        row["local_ort"] = True
        row["output_dtypes"] = [str(value.dtype) for value in values]
    except Exception as exc:
        errors.append(f"ort:{type(exc).__name__}:{exc}")
    row["error"] = " | ".join(errors)
    known_error = str(row.get("kaggle_status", "")).upper() == "ERROR"
    row["status"] = (
        "blocked_online"
        if known_error
        else "local_supported_unverified"
        if row["local_checker"] and row["strict_shape_inference"] and row["official_sanitizer"] and row["local_ort"]
        else "blocked_local"
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and execute minimal ONNX runtime capability probes.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR)
    args = parser.parse_args()
    rows = [run_probe(dict(spec), args.probe_dir) for spec in PROBES]
    payload = {
        "schema_version": 1,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "probes": rows,
    }
    from full400_safety import atomic_write_json

    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
