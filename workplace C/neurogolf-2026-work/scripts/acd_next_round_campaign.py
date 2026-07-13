from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper, shape_inference


HERE = Path(__file__).resolve()
PROJECT = HERE.parents[1]
WORKPLACE_C = PROJECT.parent
REPO_ROOT = WORKPLACE_C.parent
SCORE_DOCS = WORKPLACE_C / "score_docs"
TASK_ROOT = REPO_ROOT / "neurogolf_400_tasks" / "tasks"
ASSIGNMENTS = REPO_ROOT / "assignments" / "task_assignment_400.csv"
ARCHIVE_SCAN = SCORE_DOCS / "47_ALL399_ARCHIVE_METHOD_SCAN.csv"
DEFAULT_PARENT = Path(
    r"E:/kagglegolf/submissions/candidates/GOLF_20260713_ALL399_DIRECT_13/onnx"
)
DEFAULT_PARENT_ZIP = DEFAULT_PARENT.parent / "submission.zip"
DEFAULT_ARCHIVE = PROJECT / "data" / "external" / "neurogolf7300_archive" / "onnx"
LEDGER = SCORE_DOCS / "54_ACD_NEXT_ROUND_LEDGER.csv"
VALIDATION = SCORE_DOCS / "54_ACD_VALIDATION.csv"
COSTS = SCORE_DOCS / "54_ACD_COSTS.json"
PARENT_MANIFEST = SCORE_DOCS / "54_ACD_PARENT_MANIFEST.json"


A_P0 = {
    "task004", "task005", "task014", "task017", "task148", "task187",
    "task196", "task202", "task233", "task319", "task338", "task366",
    "task367", "task379",
}
A_P1 = {
    "task058", "task086", "task102", "task120", "task122", "task124",
    "task126", "task139", "task168", "task169", "task177", "task188",
    "task197", "task206", "task213", "task214", "task215", "task218",
    "task220", "task222", "task260", "task281", "task288", "task297",
    "task303", "task330", "task340", "task345", "task346", "task348",
    "task365", "task374", "task398",
}
C_P0 = {
    "task077", "task096", "task165", "task201", "task278", "task286",
    "task349", "task364",
}
C_P2 = {
    "task052", "task072", "task108", "task113", "task142", "task144",
    "task203", "task252", "task276", "task298", "task307", "task311",
    "task347", "task373", "task391",
}
D_P0 = {
    "task002", "task029", "task074", "task107", "task133", "task137",
    "task145", "task173", "task182", "task219", "task359", "task363",
    "task390",
}
D_P2 = {
    "task028", "task073", "task082", "task087", "task095", "task155",
    "task164", "task166", "task167", "task229", "task236", "task292",
    "task296", "task314", "task357",
}
EXPLICIT_FROZEN = {
    "task073", "task087", "task113", "task164", "task166", "task276",
    "task307", "task311",
}
D_P0_ROUTES = {
    "task002": "closed_region_gap_fill_hard_margin",
    "task029": "scalar_location_direct_gather",
    "task074": "factorized_symmetry_transform",
    "task107": "fixed_geometry_low_rank_direct_output",
    "task133": "scalar_bbox_code_region_decode",
    "task137": "winner_support_index_compression",
    "task145": "color1_color8_local_hard_margin_fusion",
    "task173": "affine_gather_scatter_where_compression",
    "task182": "winner_support_index_compression",
    "task219": "color8_neighbor_to_color1_quantized_rule",
    "task359": "winner_support_index_compression",
    "task363": "winner_support_index_compression",
    "task390": "winner_support_index_compression",
}
LEDGER_FIELDS = [
    "task", "group", "priority", "route", "status", "frozen",
    "parent_package_sha", "parent_model_sha", "parent_memory", "parent_params",
    "parent_cost", "parent_points", "parent_valid", "archive_present",
    "archive_valid", "archive_sha", "archive_memory", "archive_params",
    "archive_cost", "method_family", "archive_transfer_hint",
    "current_winner_source", "winner_in_parent", "examples_checked",
    "examples_passed", "output_shape_rule", "input_colors", "new_output_colors",
    "rule_category", "candidate_name", "candidate_sha", "candidate_memory",
    "candidate_params", "candidate_cost", "delta_cost", "delta_points",
    "accepted", "artifact_path", "provenance", "rejection_reason", "notes",
]
VALIDATION_FIELDS = [
    "task", "group", "priority", "source", "path", "sha256", "ok",
    "valid_all_checked", "examples_checked", "examples_passed",
    "examples_failed", "memory", "params", "cost", "points", "error",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    temp.replace(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def assignment_scope() -> list[dict[str, str]]:
    rows = [
        row for row in read_csv(ASSIGNMENTS)
        if row["assignment_type"] == "primary" and row["owner"] in {"A", "C", "D"}
    ]
    if len(rows) != 201 or len({row["task"] for row in rows}) != 201:
        raise RuntimeError(f"expected 201 unique primary A/C/D tasks, found {len(rows)}")
    return rows


def priority_for(group: str, task: str) -> str:
    if group == "A":
        if task in A_P0:
            return "P0"
        if task in A_P1:
            return "P1"
        return "P2"
    if group == "C":
        if task in C_P0:
            return "P0"
        if task in C_P2:
            return "P2"
        return "P1"
    if task in D_P0:
        return "P0"
    if task in D_P2:
        return "P2"
    return "P1"


def route_for(group: str, task: str, priority: str) -> str:
    if task in EXPLICIT_FROZEN:
        return "freeze_metadata_natural_lower_bound_audit"
    if group == "A":
        if priority == "P0":
            return "shape_relayout_scalar_location_direct_output"
        if priority == "P1":
            return "shape_equation_parent_aware_rebuild"
        return "shape_relayout_lower_bound_audit"
    if group == "D" and task in D_P0_ROUTES:
        return D_P0_ROUTES[task]
    if group == "D":
        return "task_specific_historical_heading_then_archive_family"
    return "parent_winner_compression_then_independent_semantic_rebuild"


def dtype_bytes(data_type: int) -> int:
    mapping = {
        TensorProto.FLOAT: 4, TensorProto.UINT8: 1, TensorProto.INT8: 1,
        TensorProto.UINT16: 2, TensorProto.INT16: 2, TensorProto.INT32: 4,
        TensorProto.INT64: 8, TensorProto.BOOL: 1, TensorProto.FLOAT16: 2,
        TensorProto.DOUBLE: 8, TensorProto.UINT32: 4, TensorProto.UINT64: 8,
        TensorProto.BFLOAT16: 2,
    }
    return mapping.get(data_type, 0)


def value_shapes(model: onnx.ModelProto) -> dict[str, dict[str, Any]]:
    try:
        inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    except Exception:
        inferred = shape_inference.infer_shapes(model)
    result: dict[str, dict[str, Any]] = {}
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    for value in values:
        tensor = value.type.tensor_type
        dims: list[int | str] = []
        known = True
        for dim in tensor.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(int(dim.dim_value))
            elif dim.HasField("dim_param"):
                dims.append(dim.dim_param)
                known = False
            else:
                dims.append("?")
                known = False
        elements = int(np.prod(dims)) if known and dims else None
        item_bytes = dtype_bytes(tensor.elem_type)
        result[value.name] = {
            "shape": dims,
            "dtype": int(tensor.elem_type),
            "elements": elements,
            "bytes": elements * item_bytes if elements is not None and item_bytes else None,
        }
    return result


def constant_elements(node: onnx.NodeProto) -> int:
    total = 0
    for attr in node.attribute:
        if attr.name == "value":
            total += int(np.prod(attr.t.dims or [1]))
        elif attr.name in {"value_floats", "value_ints", "value_strings"}:
            total += len(attr.floats or attr.ints or attr.strings)
        elif attr.name in {"value_float", "value_int", "value_string"}:
            total += 1
    return total


def graph_profile(path: Path) -> dict[str, Any]:
    model = onnx.load(str(path))
    onnx.checker.check_model(model, full_check=True)
    shapes = value_shapes(model)
    output_names = {item.name for item in model.graph.output}
    consumers: Counter[str] = Counter(name for node in model.graph.node for name in node.input)
    initializers: list[dict[str, Any]] = []
    initializer_hashes: defaultdict[str, list[str]] = defaultdict(list)
    for initializer in model.graph.initializer:
        array = numpy_helper.to_array(initializer)
        digest = hashlib.sha256(array.tobytes()).hexdigest()
        initializer_hashes[digest].append(initializer.name)
        initializers.append({
            "name": initializer.name,
            "shape": list(initializer.dims),
            "dtype": int(initializer.data_type),
            "elements": int(array.size),
            "bytes": int(array.nbytes),
            "consumers": int(consumers[initializer.name]),
        })
    nodes: list[dict[str, Any]] = []
    producer: dict[str, int] = {}
    for index, node in enumerate(model.graph.node):
        node_outputs = [shapes.get(name, {}) for name in node.output]
        max_bytes = max((item.get("bytes") or 0 for item in node_outputs), default=0)
        nodes.append({
            "index": index,
            "op_type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "output_tensors": node_outputs,
            "estimated_output_bytes": max_bytes,
            "is_final_output": any(name in output_names for name in node.output),
            "constant_elements": constant_elements(node) if node.op_type == "Constant" else 0,
            "full_space_intermediate": bool(max_bytes >= 9000 and not any(name in output_names for name in node.output)),
        })
        for name in node.output:
            producer[name] = index
    consecutive_casts = []
    for node in nodes:
        if node["op_type"] != "Cast":
            continue
        for name in node["inputs"]:
            previous = producer.get(name)
            if previous is not None and nodes[previous]["op_type"] == "Cast":
                consecutive_casts.append([previous, node["index"]])
    largest_nodes = sorted(nodes, key=lambda item: item["estimated_output_bytes"], reverse=True)[:3]
    largest_initializers = sorted(initializers, key=lambda item: item["elements"], reverse=True)[:3]
    duplicate_initializers = [names for names in initializer_hashes.values() if len(names) > 1]
    unused_initializers = [item["name"] for item in initializers if item["consumers"] == 0]
    ops = Counter(node.op_type for node in model.graph.node)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "nodes": nodes,
        "node_count": len(nodes),
        "op_types": dict(sorted(ops.items())),
        "initializers": initializers,
        "largest_nodes": largest_nodes,
        "largest_initializers": largest_initializers,
        "duplicate_initializers": duplicate_initializers,
        "unused_initializers": unused_initializers,
        "consecutive_casts": consecutive_casts,
        "full_space_intermediate_count": sum(item["full_space_intermediate"] for item in nodes),
        "terminal_op": nodes[-1]["op_type"] if nodes else "",
    }


def task_summary(task: str) -> dict[str, Any]:
    payload = json.loads((TASK_ROOT / f"{task}.json").read_text(encoding="utf-8"))
    rows = []
    input_colors: set[int] = set()
    output_colors: set[int] = set()
    input_shapes: set[str] = set()
    output_shapes: set[str] = set()
    split_counts = {}
    for split in ("train", "test", "arc-gen"):
        split_counts[split] = len(payload.get(split, []))
        for index, example in enumerate(payload.get(split, [])):
            inp = example["input"]
            out = example["output"]
            input_colors.update(value for line in inp for value in line)
            output_colors.update(value for line in out for value in line)
            ishape = f"{len(inp)}x{len(inp[0]) if inp else 0}"
            oshape = f"{len(out)}x{len(out[0]) if out else 0}"
            input_shapes.add(ishape)
            output_shapes.add(oshape)
            rows.append({"split": split, "index": index, "input_shape": ishape, "output_shape": oshape})
    same_shape = all(row["input_shape"] == row["output_shape"] for row in rows)
    if same_shape:
        rule_category = "same_shape_new_colors" if output_colors - input_colors else "same_shape_input_palette"
    else:
        rule_category = "shape_change"
    return {
        "split_counts": split_counts,
        "examples": len(rows),
        "input_shapes": sorted(input_shapes),
        "output_shapes": sorted(output_shapes),
        "input_colors": sorted(input_colors),
        "output_colors": sorted(output_colors),
        "new_output_colors": sorted(output_colors - input_colors),
        "output_shape_rule": "same_shape" if same_shape else "shape_change",
        "rule_category": rule_category,
    }


def score_job(job: tuple[str, str, str]) -> dict[str, Any]:
    task, source, raw_path = job
    path = Path(raw_path)
    sys.path.insert(0, str(HERE.parent))
    from c_score_common import score_onnx

    result = score_onnx(task, path, validate_all=True)
    try:
        profile = graph_profile(path)
    except Exception as exc:
        profile = {"path": str(path), "sha256": sha256_file(path), "profile_error": f"{type(exc).__name__}:{exc}"}
    return {"task": task, "source": source, "score": asdict(result), "profile": profile}


def archive_index() -> dict[str, dict[str, str]]:
    return {row["task"]: row for row in read_csv(ARCHIVE_SCAN)}


def validation_score(row: dict[str, str]) -> dict[str, Any]:
    integer_fields = {
        "examples_checked", "examples_passed", "examples_failed", "memory",
        "params", "cost",
    }
    score: dict[str, Any] = {
        "task": row["task"],
        "path": row["path"],
        "ok": row["ok"].lower() == "true",
        "valid_all_checked": row["valid_all_checked"].lower() == "true",
        "sha256": row["sha256"],
        "error": row.get("error", ""),
        "file_size": Path(row["path"]).stat().st_size,
    }
    for field in integer_fields:
        value = row.get(field, "")
        score[field] = int(float(value)) if value not in {"", None} else None
    value = row.get("points", "")
    score["points"] = float(value) if value not in {"", None} else None
    return score


def restored_result(row: dict[str, str]) -> dict[str, Any]:
    path = Path(row["path"])
    try:
        profile = graph_profile(path)
    except Exception as exc:
        profile = {
            "path": str(path), "sha256": row["sha256"],
            "profile_error": f"{type(exc).__name__}:{exc}",
        }
    return {
        "task": row["task"],
        "source": row["source"],
        "score": validation_score(row),
        "profile": profile,
    }


def seed(args: argparse.Namespace) -> None:
    parent_dir = args.parent_dir.resolve()
    archive_dir = args.archive_dir.resolve()
    parent_files = sorted(parent_dir.glob("task*.onnx"))
    expected = [f"task{index:03d}" for index in range(1, 401)]
    if [path.stem for path in parent_files] != expected:
        raise RuntimeError("parent must contain exactly task001-task400 ONNX files")
    if not args.parent_zip.exists():
        raise FileNotFoundError(args.parent_zip)
    package_sha = sha256_file(args.parent_zip)
    scope_rows = assignment_scope()
    scope = {row["task"]: row for row in scope_rows}
    archive_rows = archive_index()
    all_jobs = [(task, "parent", str(parent_dir / f"{task}.onnx")) for task in expected]
    for task in sorted(scope):
        archive_path = archive_dir / f"{task}.onnx"
        if archive_path.exists():
            all_jobs.append((task, "archive", str(archive_path)))

    completed: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    validation_rows: list[dict[str, Any]] = []
    if args.resume and VALIDATION.exists():
        restored_rows = read_csv(VALIDATION)
        allowed = {(task, source) for task, source, _ in all_jobs}
        seen: set[tuple[str, str]] = set()
        for row in restored_rows:
            key = (row.get("task", ""), row.get("source", ""))
            if key not in allowed or key in seen:
                continue
            seen.add(key)
            result = restored_result(row)
            completed[key[0]][key[1]] = result
            validation_rows.append(row)
        print(json.dumps({"restored": len(validation_rows), "total": len(all_jobs)}), flush=True)
    jobs = [job for job in all_jobs if job[1] not in completed[job[0]]]
    costs: dict[str, Any] = {
        "parent": {
            "path": str(parent_dir),
            "zip": str(args.parent_zip.resolve()),
            "package_sha256": package_sha,
            "public_ref": args.public_ref,
            "public_score": args.public_score,
        },
        "tasks": {},
    }

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(score_job, job): job for job in jobs}
        restored_count = len(validation_rows)
        for count, future in enumerate(as_completed(futures), start=restored_count + 1):
            task, source, raw_path = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "task": task,
                    "source": source,
                    "score": {
                        "task": task, "path": raw_path, "ok": False,
                        "valid_all_checked": True, "examples_checked": 0,
                        "examples_passed": 0, "examples_failed": 0,
                        "memory": None, "params": None, "cost": None,
                        "points": None, "file_size": Path(raw_path).stat().st_size,
                        "sha256": sha256_file(Path(raw_path)),
                        "error": f"{type(exc).__name__}:{exc}",
                    },
                    "profile": {},
                }
            completed[task][source] = result
            score = result["score"]
            assignment = scope.get(task, {})
            group = assignment.get("owner", "")
            priority = priority_for(group, task) if group else ""
            validation_rows.append({
                "task": task, "group": group, "priority": priority,
                "source": source, **score,
            })
            validation_rows.sort(key=lambda row: (row["task"], row["source"]))
            write_csv(VALIDATION, validation_rows, VALIDATION_FIELDS)
            print(json.dumps({
                "completed": count, "total": len(all_jobs), "task": task,
                "source": source, "ok": score.get("ok"), "cost": score.get("cost"),
                "examples": score.get("examples_checked"),
            }), flush=True)

    ledger_rows = []
    for task in sorted(scope, key=lambda item: (priority_for(scope[item]["owner"], item), scope[item]["owner"], item)):
        assignment = scope[task]
        group = assignment["owner"]
        priority = priority_for(group, task)
        summary = task_summary(task)
        parent = completed[task]["parent"]
        archive = completed[task].get("archive")
        old_archive = archive_rows.get(task, {})
        parent_score = parent["score"]
        archive_score = archive["score"] if archive else {}
        dynamic_frozen = priority == "P2" and parent_score.get("cost") is not None and parent_score["cost"] <= 30
        frozen = task in EXPLICIT_FROZEN or dynamic_frozen
        ledger_rows.append({
            "task": task,
            "group": group,
            "priority": priority,
            "route": route_for(group, task, priority),
            "status": "frozen_audit_pending" if frozen else "profiled_pending_candidates",
            "frozen": str(frozen).lower(),
            "parent_package_sha": package_sha,
            "parent_model_sha": parent_score.get("sha256", ""),
            "parent_memory": parent_score.get("memory", ""),
            "parent_params": parent_score.get("params", ""),
            "parent_cost": parent_score.get("cost", ""),
            "parent_points": parent_score.get("points", ""),
            "parent_valid": str(bool(parent_score.get("ok"))).lower(),
            "archive_present": str(bool(archive)).lower(),
            "archive_valid": str(bool(archive_score.get("ok"))).lower() if archive else "false",
            "archive_sha": archive_score.get("sha256", ""),
            "archive_memory": archive_score.get("memory", ""),
            "archive_params": archive_score.get("params", ""),
            "archive_cost": archive_score.get("cost", ""),
            "method_family": old_archive.get("archive_method_family", "missing_archive"),
            "archive_transfer_hint": old_archive.get("transfer_hint", "independent_parent_semantic_rebuild"),
            "current_winner_source": "batch13_parent",
            "winner_in_parent": "true",
            "examples_checked": parent_score.get("examples_checked", ""),
            "examples_passed": parent_score.get("examples_passed", ""),
            "output_shape_rule": summary["output_shape_rule"],
            "input_colors": ";".join(map(str, summary["input_colors"])),
            "new_output_colors": ";".join(map(str, summary["new_output_colors"])),
            "rule_category": summary["rule_category"],
            "accepted": "false",
            "provenance": "batch13_parent_remeasured",
            "notes": assignment.get("note", ""),
        })
        costs["tasks"][task] = {
            "group": group,
            "priority": priority,
            "route": route_for(group, task, priority),
            "task_summary": summary,
            "parent": parent,
            "archive": archive,
        }

    parent_validation = [row for row in validation_rows if row["source"] == "parent"]
    parent_ok = len(parent_validation) == 400 and all(str(row["ok"]).lower() == "true" for row in parent_validation)
    manifest = {
        "parent_dir": str(parent_dir),
        "parent_zip": str(args.parent_zip.resolve()),
        "parent_package_sha256": package_sha,
        "public_ref": args.public_ref,
        "public_score": args.public_score,
        "root_onnx_count": len(parent_files),
        "parent_models_full_valid": sum(str(row["ok"]).lower() == "true" for row in parent_validation),
        "parent_models_total": len(parent_validation),
        "parent_full_validation_ok": parent_ok,
        "parent_examples_checked": sum(int(row.get("examples_checked") or 0) for row in parent_validation),
        "scope_counts": dict(Counter(row["group"] for row in ledger_rows)),
        "priority_counts": dict(Counter(row["priority"] for row in ledger_rows)),
        "frozen_count": sum(row["frozen"] == "true" for row in ledger_rows),
    }
    write_csv(LEDGER, ledger_rows, LEDGER_FIELDS)
    write_json(COSTS, costs)
    write_json(PARENT_MANIFEST, manifest)
    print(json.dumps(manifest, indent=2))
    if not parent_ok:
        raise SystemExit("parent full validation failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["seed"])
    parser.add_argument("--parent-dir", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--parent-zip", type=Path, default=DEFAULT_PARENT_ZIP)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--public-ref", default="54638601")
    parser.add_argument("--public-score", type=float, default=7378.01)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage == "seed":
        seed(args)


if __name__ == "__main__":
    main()
