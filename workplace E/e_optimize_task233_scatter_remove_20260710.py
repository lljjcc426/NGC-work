#!/usr/bin/env python
"""Remove task233 external patches with their known indices via ScatterElements."""
from __future__ import annotations

import copy
import csv
import importlib.util
import math
import pathlib
import zipfile

import numpy as np
import onnx
from onnx import helper, numpy_helper


WORKPLACE = pathlib.Path(__file__).resolve().parent
NGC_ROOT = pathlib.Path(r"F:\kaggle\neurogolf-2026")
BASE_ZIP = NGC_ROOT / "submissions" / "submission.zip"
OUT_DIR = WORKPLACE / "optimized_onnx" / "task233_scatter_remove_20260710"
OUT_CSV = WORKPLACE / "e_task233_scatter_remove_20260710.csv"
HELPER_PATH = WORKPLACE / "e_optimize_task233_yusuke_20260710.py"


def load_helpers():
    spec = importlib.util.spec_from_file_location("e_task233_helpers", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load task233 helper module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def optimize(base: onnx.ModelProto) -> onnx.ModelProto:
    model = copy.deepcopy(base)
    nodes = list(model.graph.node)
    remove_outputs = {"safe_name_71", "safe_name_72", "safe_name_73", "safe_name_74"}
    remove_indices = [
        index
        for index, node in enumerate(nodes)
        if any(output in remove_outputs for output in node.output)
    ]
    if remove_indices != [22, 23, 24, 25]:
        raise RuntimeError(f"unexpected removal branch: {remove_indices}")

    replacement = [
        helper.make_node(
            "Where",
            ["safe_name_66", "safe_name_29", "safe_name_29"],
            ["task233_zero_patch_updates"],
            name="task233_zero_patch_updates",
        ),
        helper.make_node(
            "Where",
            ["safe_name_58", "safe_name_63", "task233_invalid_patch_index"],
            ["task233_patch_indices"],
            name="task233_patch_indices",
        ),
        helper.make_node(
            "Reshape",
            ["task233_patch_indices", "task233_patch_flat_shape"],
            ["task233_patch_indices_flat"],
            name="task233_patch_indices_flat",
        ),
        helper.make_node(
            "Reshape",
            ["task233_zero_patch_updates", "task233_patch_flat_shape"],
            ["task233_zero_patch_updates_flat"],
            name="task233_zero_patch_updates_flat",
        ),
        helper.make_node(
            "ScatterElements",
            [
                "safe_name_64",
                "task233_patch_indices_flat",
                "task233_zero_patch_updates_flat",
            ],
            ["task233_main_flat"],
            name="task233_main_flat",
            axis=0,
        ),
        helper.make_node(
            "Reshape",
            ["task233_main_flat", "task233_grid_shape"],
            ["safe_name_74"],
            name="safe_name_74",
        ),
    ]
    nodes[22:26] = replacement
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.graph.initializer.append(
        numpy_helper.from_array(
            np.asarray(899, dtype=np.int32),
            name="task233_invalid_patch_index",
        )
    )
    model.graph.initializer.append(
        numpy_helper.from_array(
            np.asarray([45], dtype=np.int64),
            name="task233_patch_flat_shape",
        )
    )
    model.graph.initializer.append(
        numpy_helper.from_array(
            np.asarray([1, 1, 30, 30], dtype=np.int64),
            name="task233_grid_shape",
        )
    )
    kept_value_info = [
        item
        for item in model.graph.value_info
        if item.name not in {"safe_name_71", "safe_name_72", "safe_name_73"}
    ]
    kept_value_info.extend(
        [
            helper.make_tensor_value_info(
                "task233_zero_patch_updates", onnx.TensorProto.UINT8, [5, 9]
            ),
            helper.make_tensor_value_info(
                "task233_patch_indices", onnx.TensorProto.INT32, [5, 9]
            ),
            helper.make_tensor_value_info(
                "task233_patch_indices_flat", onnx.TensorProto.INT32, [45]
            ),
            helper.make_tensor_value_info(
                "task233_zero_patch_updates_flat", onnx.TensorProto.UINT8, [45]
            ),
            helper.make_tensor_value_info(
                "task233_main_flat", onnx.TensorProto.UINT8, [900]
            ),
        ]
    )
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_value_info)
    onnx.checker.check_model(model, full_check=True)
    return model


def points(cost: int) -> float:
    return max(1.0, 25.0 - math.log(cost))


def main() -> int:
    helpers = load_helpers()
    with zipfile.ZipFile(BASE_ZIP) as archive:
        base = onnx.load_from_string(archive.read("task233.onnx"))
    candidate = optimize(base)

    base_cost = helpers.score_cost(base)
    candidate_cost = helpers.score_cost(candidate)
    sample_ok = candidate_cost is not None and helpers.sample_ok(candidate)
    full_ok = sample_ok and helpers.verify_all(candidate)
    accepted = (
        base_cost is not None
        and candidate_cost is not None
        and candidate_cost < base_cost
        and full_ok
    )
    row = {
        "task": "task233",
        "candidate": "scatter_remove_external_patches",
        "status": "ok_improved" if accepted else "not_accepted",
        "arc_agi_pass": 4 if full_ok else "",
        "arc_gen_pass": 262 if full_ok else "",
        "base_cost": base_cost,
        "candidate_cost": candidate_cost,
        "delta_cost": "" if candidate_cost is None else base_cost - candidate_cost,
        "base_points": "" if base_cost is None else f"{points(base_cost):.9f}",
        "candidate_points": "" if candidate_cost is None else f"{points(candidate_cost):.9f}",
        "delta_points": (
            ""
            if base_cost is None or candidate_cost is None
            else f"{points(candidate_cost) - points(base_cost):.9f}"
        ),
        "nodes": len(candidate.graph.node),
        "initializers": len(candidate.graph.initializer),
    }
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    if accepted:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        onnx.save(candidate, OUT_DIR / "task233.onnx")
    print(row)
    print(f"Wrote {OUT_CSV}")
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
