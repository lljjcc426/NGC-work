#!/usr/bin/env python
"""Replace task233's descending-rank tensor with an exact masked TopK."""
from __future__ import annotations

import copy
import csv
import importlib.util
import math
import pathlib

import onnx
from onnx import TensorProto, helper


WORKPLACE = pathlib.Path(__file__).resolve().parent
BASE = (
    WORKPLACE
    / "optimized_onnx"
    / "task233_scatter_remove_20260710"
    / "task233.onnx"
)
OUT_DIR = WORKPLACE / "optimized_onnx" / "task233_masked_topk_20260710"
OUT_MODEL = OUT_DIR / "task233.onnx"
OUT_CSV = WORKPLACE / "e_task233_masked_topk_20260710.csv"
HELPERS_PATH = WORKPLACE / "e_optimize_task233_yusuke_20260710.py"


def load_helpers():
    spec = importlib.util.spec_from_file_location("task233_helpers", HELPERS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load task233 helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prune(model: onnx.ModelProto) -> None:
    needed = {output.name for output in model.graph.output}
    kept_reversed = []
    for node in reversed(model.graph.node):
        if any(output in needed for output in node.output):
            kept_reversed.append(node)
            needed.update(value for value in node.input if value)
    kept_ids = {id(node) for node in kept_reversed}
    kept = [node for node in model.graph.node if id(node) in kept_ids]
    used = {value for node in kept for value in node.input if value}
    kept_initializers = [init for init in model.graph.initializer if init.name in used]
    del model.graph.node[:]
    model.graph.node.extend(kept)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)


def optimize(base: onnx.ModelProto) -> onnx.ModelProto:
    model = copy.deepcopy(base)
    replacements = [
        helper.make_node(
            "Where",
            ["safe_name_99", "safe_name_104", "safe_name_24"],
            ["task233_masked_codes"],
            name="task233_mask_codes",
        ),
        helper.make_node(
            "Equal",
            ["task233_masked_codes", "safe_name_70"],
            ["safe_name_105"],
            name="task233_valid_matches",
        ),
        helper.make_node(
            "Cast",
            ["safe_name_105"],
            ["safe_name_106"],
            name="task233_valid_matches_f16",
            to=TensorProto.FLOAT16,
        ),
        helper.make_node(
            "TopK",
            ["safe_name_106", "safe_name_9"],
            ["safe_name_107", "safe_name_108"],
            name="task233_first_two_valid_indices",
            axis=-1,
            largest=1,
            sorted=1,
        ),
        helper.make_node(
            "Cast",
            ["safe_name_107"],
            ["safe_name_110"],
            name="task233_has_valid_match",
            to=TensorProto.BOOL,
        ),
    ]
    remove_outputs = {"safe_name_100", "safe_name_106", "safe_name_107", "safe_name_108", "safe_name_110"}
    new_nodes = []
    inserted = False
    for node in model.graph.node:
        if "safe_name_105" in node.output:
            new_nodes.extend(replacements)
            inserted = True
            continue
        if any(output in remove_outputs for output in node.output):
            continue
        new_nodes.append(node)
    if not inserted:
        raise RuntimeError("task233 masked TopK block was not found")
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    prune(model)
    live_outputs = {output for node in model.graph.node for output in node.output if output}
    kept_value_info = [
        value_info for value_info in model.graph.value_info if value_info.name in live_outputs
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_value_info)
    model.graph.value_info.append(
        helper.make_tensor_value_info(
            "task233_masked_codes", TensorProto.FLOAT16, [1, 324]
        )
    )
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    return model


def points(cost: int) -> float:
    return max(1.0, 25.0 - math.log(max(1, cost)))


def main() -> int:
    helpers = load_helpers()
    base = onnx.load(BASE)
    candidate = optimize(base)
    base_cost = helpers.score_cost(base)
    candidate_cost = helpers.score_cost(candidate)
    verified_all = candidate_cost is not None and helpers.verify_all(candidate)
    status = (
        "ok_improved"
        if base_cost is not None
        and candidate_cost is not None
        and candidate_cost < base_cost
        and verified_all
        else "rejected"
    )
    if status == "ok_improved":
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        onnx.save(candidate, OUT_MODEL)

    row = {
        "task": "task233",
        "candidate": "masked_topk",
        "status": status,
        "verified_all": verified_all,
        "base_cost": base_cost,
        "candidate_cost": candidate_cost,
        "delta_cost": "" if candidate_cost is None or base_cost is None else base_cost - candidate_cost,
        "base_points": "" if base_cost is None else f"{points(base_cost):.9f}",
        "candidate_points": "" if candidate_cost is None else f"{points(candidate_cost):.9f}",
        "delta_points": "" if candidate_cost is None or base_cost is None else f"{points(candidate_cost) - points(base_cost):.9f}",
        "nodes": len(candidate.graph.node),
        "initializers": len(candidate.graph.initializer),
        "output": str(OUT_MODEL) if status == "ok_improved" else "",
    }
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(row)
    return 0 if status == "ok_improved" else 1


if __name__ == "__main__":
    raise SystemExit(main())
