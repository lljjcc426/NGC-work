from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve()
PROJECT = HERE.parents[1]
WORKPLACE_C = PROJECT.parent
DEFAULT_ARCHIVE = PROJECT / "data" / "external" / "neurogolf7300_archive" / "onnx"
DEFAULT_PARENT = Path(
    r"E:/kagglegolf/submissions/candidates/GOLF_20260713_C5_05/onnx"
)
DEFAULT_OUTPUT = WORKPLACE_C / "score_docs" / "47_ALL399_ARCHIVE_METHOD_SCAN.csv"


def constant_elements(model: onnx.ModelProto) -> int:
    total = 0
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value":
                total += int(np.prod(attr.t.dims or [1]))
            elif attr.name == "sparse_value":
                total += int(np.prod(attr.sparse_tensor.values.dims or [1]))
            elif attr.name in {"value_floats", "value_ints", "value_strings"}:
                total += len(attr.floats or attr.ints or attr.strings)
            elif attr.name in {"value_float", "value_int", "value_string"}:
                total += 1
    return total


def graph_profile(path: Path) -> dict[str, object]:
    model = onnx.load(str(path))
    ops = Counter(node.op_type for node in model.graph.node)
    initializer_elements = sum(
        int(np.prod(item.dims or [1])) for item in model.graph.initializer
    )
    constants = constant_elements(model)
    terminal = model.graph.node[-1].op_type if model.graph.node else ""
    return {
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "initializer_elements": initializer_elements,
        "constant_elements": constants,
        "parameter_elements_static": initializer_elements + constants,
        "terminal_op": terminal,
        "op_types": ";".join(f"{name}:{count}" for name, count in sorted(ops.items())),
        "opset": max((item.version for item in model.opset_import if item.domain in {"", "ai.onnx"}), default=0),
        "file_size": path.stat().st_size,
    }


def method_family(profile: dict[str, object]) -> str:
    ops = {
        item.split(":", 1)[0]
        for item in str(profile["op_types"]).split(";")
        if item
    }
    nodes = int(profile["nodes"])
    terminal = str(profile["terminal_op"])
    if nodes == 1 and terminal == "Einsum":
        return "single_terminal_einsum"
    if terminal == "Einsum":
        return "terminal_einsum"
    if "QLinearConv" in ops:
        return "quantized_local_detection"
    if "Conv" in ops:
        return "conv_local_detection"
    if any(name.startswith("Bitwise") for name in ops):
        return "bitpacked_state_logic"
    if {"Gather", "GatherElements", "GatherND", "ScatterElements", "ScatterND"} & ops:
        return "gather_scatter_relayout"
    if {"MaxPool", "AveragePool", "GlobalMaxPool", "GlobalAveragePool"} & ops:
        return "pool_propagation"
    if any(name.startswith("Reduce") for name in ops):
        return "reduction_mask_logic"
    if "Resize" in ops:
        return "resize_shape_transform"
    if nodes <= 3:
        return "small_direct_graph"
    return "general_tensor_graph"


def transfer_hint(archive: dict[str, object], parent: dict[str, object]) -> str:
    family = method_family(archive)
    if family == "single_terminal_einsum":
        return "rederive_rule_as_one_direct_output_einsum"
    if family == "terminal_einsum":
        return "fuse_shape_or_mask_selection_into_terminal_einsum"
    if family == "quantized_local_detection":
        return "derive_integer_margin_and_crop_local_kernel_support"
    if family == "conv_local_detection":
        return "derive_minimal_local_kernel_and_remove_zero_halo"
    if family == "bitpacked_state_logic":
        return "pack_boolean_state_before_spatial_materialization"
    if family == "gather_scatter_relayout":
        return "derive_direct_coordinate_mapping_or_terminal_gather"
    if int(archive["nodes"]) < int(parent["nodes"]):
        return "rederive_rule_with_fewer_materialized_intermediates"
    return "inspect_constant_and_shape_control_compression"


def run_task(job: tuple[str, str, str]) -> dict[str, object]:
    task, archive_dir_raw, parent_dir_raw = job
    archive_path = Path(archive_dir_raw) / f"{task}.onnx"
    parent_path = Path(parent_dir_raw) / f"{task}.onnx"
    archive_profile = graph_profile(archive_path)
    parent_profile = graph_profile(parent_path)
    row: dict[str, object] = {
        "task": task,
        "archive_path": str(archive_path),
        "parent_path": str(parent_path),
        "archive_method_family": method_family(archive_profile),
        "transfer_hint": transfer_hint(archive_profile, parent_profile),
        "source_policy": "analysis_only_do_not_copy_or_submit_archive_model",
    }
    row.update({f"archive_{key}": value for key, value in archive_profile.items()})
    row.update({f"parent_{key}": value for key, value in parent_profile.items()})

    sys.path.insert(0, str(HERE.parent))
    from c_score_common import score_onnx

    archive_score = score_onnx(task, archive_path, True)
    parent_score = score_onnx(task, parent_path, True)
    row.update({f"archive_score_{key}": value for key, value in asdict(archive_score).items()})
    row.update({f"parent_score_{key}": value for key, value in asdict(parent_score).items()})
    row["archive_minus_parent_cost"] = (
        archive_score.cost - parent_score.cost
        if archive_score.cost is not None and parent_score.cost is not None
        else ""
    )
    row["archive_lower_and_valid"] = bool(
        archive_score.ok
        and parent_score.ok
        and archive_score.cost is not None
        and parent_score.cost is not None
        and archive_score.cost < parent_score.cost
    )
    return row


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--parent-dir", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    tasks = sorted(path.stem for path in args.archive_dir.glob("task*.onnx"))
    if len(tasks) != 399:
        raise RuntimeError(f"expected 399 archive models, found {len(tasks)}")
    missing = [f"task{index:03d}" for index in range(1, 401) if f"task{index:03d}" not in tasks]

    rows: list[dict[str, object]] = []
    jobs = [(task, str(args.archive_dir.resolve()), str(args.parent_dir.resolve())) for task in tasks]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_task, job): job[0] for job in jobs}
        for index, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "task": task,
                    "status": "worker_failed",
                    "error": f"{type(exc).__name__}:{exc}",
                    "source_policy": "analysis_only_do_not_copy_or_submit_archive_model",
                }
            rows.append(row)
            rows.sort(key=lambda item: str(item["task"]))
            write_rows(args.output, rows)
            print(
                json.dumps(
                    {
                        "completed": index,
                        "task": task,
                        "archive_lower": row.get("archive_lower_and_valid", ""),
                    }
                ),
                flush=True,
            )

    families = Counter(str(row.get("archive_method_family")) for row in rows)
    summary = {
        "archive_models": len(rows),
        "missing_tasks": missing,
        "archive_valid": sum(str(row.get("archive_score_ok")).lower() == "true" for row in rows),
        "parent_valid": sum(str(row.get("parent_score_ok")).lower() == "true" for row in rows),
        "archive_lower_and_valid": sum(bool(row.get("archive_lower_and_valid")) for row in rows),
        "method_families": dict(sorted(families.items())),
        "source_policy": "analysis_only_do_not_copy_or_submit_archive_model",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
