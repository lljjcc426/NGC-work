from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto

from full400_safety import ALLOWED_CANDIDATE_STATES, atomic_write_json, read_json, sha256_file


HERE = Path(__file__).resolve()
PROJECT = HERE.parent.parent
DEFAULT_REGISTRY = PROJECT / "config" / "candidate_registry.json"
FORBIDDEN_TERMINAL_STATES = {"blocked", "local_only", "rejected"}
PROGRESS_STATES = (
    "experimental",
    "checker_passed",
    "official_full_passed",
    "fuzz_passed",
    "generator_passed",
    "runtime_safe",
    "canonical",
    "online_verified",
)
ALL_STATES = set(PROGRESS_STATES) | FORBIDDEN_TERMINAL_STATES


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def empty_registry() -> dict[str, Any]:
    return {"schema_version": 1, "updated_at": utc_now(), "candidates": []}


def load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_registry()
    payload = read_json(path)
    if payload.get("schema_version") != 1 or not isinstance(payload.get("candidates"), list):
        raise RuntimeError(f"invalid candidate registry: {path}")
    return payload


def _value_types(model: onnx.ModelProto) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        result[value.name] = value.type.tensor_type.elem_type
    for tensor in model.graph.initializer:
        result[tensor.name] = tensor.data_type
    return result


def operator_audit(path: Path) -> dict[str, Any]:
    model = onnx.load(path)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    types = _value_types(inferred)
    forbidden_ops: list[str] = []
    negative_padding: list[dict[str, Any]] = []
    operators: dict[str, int] = {}
    for index, node in enumerate(inferred.graph.node):
        operators[node.op_type] = operators.get(node.op_type, 0) + 1
        if node.op_type == "TopK" and node.input and types.get(node.input[0]) == TensorProto.UINT8:
            forbidden_ops.append(f"node_{index}:TopK:uint8")
        if node.op_type in {"Conv", "QLinearConv"}:
            pads = next((attr for attr in node.attribute if attr.name == "pads"), None)
            if pads is not None and any(value < 0 for value in pads.ints):
                negative_padding.append(
                    {"node": node.name or f"node_{index}", "pads": list(pads.ints)}
                )
    return {
        "operators": dict(sorted(operators.items())),
        "forbidden_ops": forbidden_ops,
        "negative_padding": negative_padding,
        "runtime_compatible": not forbidden_ops and not negative_padding,
    }


def register_candidate(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    registry = load_registry(path)
    required = {"task", "candidate_path", "parent_path", "method", "status"}
    missing = sorted(required - set(record))
    if missing:
        raise RuntimeError(f"candidate record missing fields: {missing}")
    status = record["status"]
    if status not in ALL_STATES:
        raise RuntimeError(f"invalid candidate status: {status}")
    candidate = Path(record["candidate_path"])
    parent = Path(record["parent_path"])
    if not candidate.is_file() or not parent.is_file():
        raise FileNotFoundError(f"candidate or parent missing: {candidate}, {parent}")
    audit = operator_audit(candidate)
    if (audit["forbidden_ops"] or audit["negative_padding"]) and status not in FORBIDDEN_TERMINAL_STATES:
        raise RuntimeError("unsafe candidate cannot be registered as promotable")
    candidate_sha = sha256_file(candidate)
    parent_sha = sha256_file(parent)
    now = utc_now()
    existing = next(
        (
            item
            for item in registry["candidates"]
            if item.get("task") == record["task"]
            and item.get("candidate_sha256") == candidate_sha
        ),
        None,
    )
    payload = {
        "task": record["task"],
        "owner": record.get("owner", ""),
        "candidate_path": str(candidate.resolve()),
        "candidate_sha256": candidate_sha,
        "parent_path": str(parent.resolve()),
        "parent_sha256": parent_sha,
        "method": record["method"],
        "status": status,
        "risk_level": record.get("risk_level", "unknown"),
        "official_examples_checked": int(record.get("official_examples_checked", 0)),
        "official_examples_passed": int(record.get("official_examples_passed", 0)),
        "fuzz_trials": int(record.get("fuzz_trials", 0)),
        "generator_trials": int(record.get("generator_trials", 0)),
        "runtime_compatible": bool(audit["runtime_compatible"]),
        "forbidden_ops": audit["forbidden_ops"],
        "negative_padding": audit["negative_padding"],
        "public_example_selected": bool(record.get("public_example_selected", False)),
        "cost": record.get("cost"),
        "points": record.get("points"),
        "delta_cost": record.get("delta_cost"),
        "delta_points": record.get("delta_points"),
        "created_by": record.get("created_by", "candidate_registry.py"),
        "created_at": record.get("created_at", now),
        "updated_at": now,
        "notes": record.get("notes", ""),
        "operator_audit": audit,
    }
    history_entry = {"at": now, "status": status, "reason": record.get("reason", "registered")}
    if existing is None:
        payload["history"] = [history_entry]
        registry["candidates"].append(payload)
    else:
        previous = existing.get("status", "experimental")
        if previous in FORBIDDEN_TERMINAL_STATES and status not in FORBIDDEN_TERMINAL_STATES:
            raise RuntimeError(f"terminal candidate cannot be promoted: {previous} -> {status}")
        if previous in PROGRESS_STATES and status in PROGRESS_STATES:
            if PROGRESS_STATES.index(status) < PROGRESS_STATES.index(previous):
                raise RuntimeError(f"candidate status regression: {previous} -> {status}")
        payload["history"] = list(existing.get("history", [])) + [history_entry]
        existing.clear()
        existing.update(payload)
    registry["candidates"].sort(key=lambda item: (item["task"], item["candidate_sha256"]))
    registry["updated_at"] = now
    atomic_write_json(path, registry)
    return payload


def eligible_candidates(registry_path: Path, parent_hashes: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
    registry = load_registry(registry_path)
    result: dict[str, list[dict[str, Any]]] = {}
    for item in registry["candidates"]:
        if item.get("status") not in ALLOWED_CANDIDATE_STATES:
            continue
        task = item["task"]
        if item.get("parent_sha256") != parent_hashes.get(task):
            continue
        candidate = Path(item["candidate_path"])
        if not candidate.is_file() or sha256_file(candidate) != item.get("candidate_sha256"):
            continue
        if item.get("forbidden_ops") or item.get("negative_padding"):
            continue
        result.setdefault(task, []).append(item)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register and audit NeuroGolf ONNX candidates.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    validate = sub.add_parser("validate")
    validate.add_argument("--strict", action="store_true")
    register = sub.add_parser("register")
    register.add_argument("--record", type=Path, required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--model", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "init":
        if not args.registry.exists():
            atomic_write_json(args.registry, empty_registry())
        print(json.dumps(load_registry(args.registry), indent=2))
        return 0
    if args.command == "audit":
        print(json.dumps(operator_audit(args.model), indent=2))
        return 0
    if args.command == "register":
        record = read_json(args.record)
        print(json.dumps(register_candidate(args.registry, record), indent=2))
        return 0
    registry = load_registry(args.registry)
    failures = []
    for item in registry["candidates"]:
        if item.get("status") not in ALL_STATES:
            failures.append({"task": item.get("task"), "error": "invalid_status"})
        if item.get("status") in ALLOWED_CANDIDATE_STATES:
            if item.get("forbidden_ops") or item.get("negative_padding"):
                failures.append({"task": item.get("task"), "error": "unsafe_promotable"})
    print(json.dumps({"records": len(registry["candidates"]), "failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
