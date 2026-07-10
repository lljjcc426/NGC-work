#!/usr/bin/env python
"""Scan shape-compatible node bypasses for the current task233 model."""
from __future__ import annotations

import copy
import csv
import importlib.util
import pathlib
import zipfile

import onnx


WORKPLACE = pathlib.Path(__file__).resolve().parent
NGC_ROOT = pathlib.Path(r"F:\kaggle\neurogolf-2026")
BASE_ZIP = NGC_ROOT / "submissions" / "submission.zip"
OUT_CSV = WORKPLACE / "e_task233_general_bypass_20260710.csv"
OUT_DIR = WORKPLACE / "optimized_onnx" / "task233_general_bypass_20260710"
HELPER_PATH = WORKPLACE / "e_optimize_task233_yusuke_20260710.py"


def load_helpers():
    spec = importlib.util.spec_from_file_location("e_task233_helpers", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load task233 helper module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor_types(model: onnx.ModelProto) -> dict[str, tuple[tuple[int, ...], int]]:
    graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    result: dict[str, tuple[tuple[int, ...], int]] = {}
    for item in list(graph.input) + list(graph.value_info) + list(graph.output):
        if not item.type.HasField("tensor_type"):
            continue
        tensor_type = item.type.tensor_type
        shape = tuple(dim.dim_value for dim in tensor_type.shape.dim)
        result[item.name] = (shape, tensor_type.elem_type)
    return result


def candidates(base: onnx.ModelProto, helpers):
    types = tensor_types(base)
    seen: set[bytes] = set()
    for node_index, node in enumerate(base.graph.node):
        if not node.output or not node.output[0]:
            continue
        output_name = node.output[0]
        output_type = types.get(output_name)
        if output_type is None:
            continue
        for input_index, input_name in enumerate(node.input):
            if not input_name or types.get(input_name) != output_type:
                continue
            model = copy.deepcopy(base)
            old_output = model.graph.node[node_index].output[0]
            replacement = model.graph.node[node_index].input[input_index]
            for other in model.graph.node:
                for position, value in enumerate(other.input):
                    if value == old_output:
                        other.input[position] = replacement
            helpers.prune(model)
            del model.graph.value_info[:]
            payload = model.SerializeToString()
            if payload in seen:
                continue
            seen.add(payload)
            yield f"n{node_index:03d}_{node.op_type}_i{input_index}", model


def main() -> int:
    helpers = load_helpers()
    with zipfile.ZipFile(BASE_ZIP) as archive:
        base = onnx.load_from_string(archive.read("task233.onnx"))
    base_cost = helpers.score_cost(base)
    if base_cost is None:
        raise RuntimeError("could not score base model")

    rows: list[dict[str, object]] = []
    accepted: list[tuple[str, onnx.ModelProto, int]] = []
    for name, model in candidates(base, helpers):
        status = "not_checked"
        cost = helpers.score_cost(model)
        full_ok = False
        if cost is None:
            status = "score_error"
        elif cost >= base_cost:
            status = "not_better"
        elif not helpers.sample_ok(model):
            status = "sample_wrong"
        else:
            full_ok = helpers.verify_all(model)
            status = "ok_improved" if full_ok else "full_wrong"
            if full_ok:
                accepted.append((name, model, cost))
        rows.append(
            {
                "candidate": name,
                "base_cost": base_cost,
                "candidate_cost": "" if cost is None else cost,
                "delta_cost": "" if cost is None else base_cost - cost,
                "status": status,
                "verified_all": full_ok,
                "nodes": len(model.graph.node),
                "initializers": len(model.graph.initializer),
            }
        )
        print(f"{name}: status={status} cost={cost}", flush=True)

    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if accepted:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for name, model, _ in accepted:
            onnx.save(model, OUT_DIR / f"{name}.onnx")
    print(f"Wrote {OUT_CSV}")
    print(f"accepted={len(accepted)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
