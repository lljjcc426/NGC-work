from __future__ import annotations

import argparse
from collections import Counter
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
FORBIDDEN_TERMINAL_STATES = {
    "blocked",
    "blocked_hidden_distribution",
    "blocked_runtime",
    "local_only",
    "local_only_public_fit",
    "rejected",
    "rejected_mathematical",
    "rejected_no_cost_gain",
}
ARCHIVE_STATES = {"absorbed_into_baseline"}
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
ALL_STATES = set(PROGRESS_STATES) | FORBIDDEN_TERMINAL_STATES | ARCHIVE_STATES
PROMOTABLE_STATES = {"runtime_safe", "canonical", "online_verified"}
INTEGER_TYPES = {
    TensorProto.INT8,
    TensorProto.INT16,
    TensorProto.INT32,
    TensorProto.INT64,
    TensorProto.UINT8,
    TensorProto.UINT16,
    TensorProto.UINT32,
    TensorProto.UINT64,
}
TYPE_NAMES = {
    value: name.lower()
    for name, value in TensorProto.DataType.items()
}


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
    runtime_risks: list[str] = []
    negative_padding: list[dict[str, Any]] = []
    operators: dict[str, int] = {}
    for index, node in enumerate(inferred.graph.node):
        operators[node.op_type] = operators.get(node.op_type, 0) + 1
        input_type = types.get(node.input[0]) if node.input else None
        input_type_name = TYPE_NAMES.get(input_type, str(input_type))
        if node.op_type == "TopK" and input_type in {TensorProto.UINT8, TensorProto.INT8}:
            risk = f"TopK:{input_type_name}"
            forbidden_ops.append(f"node_{index}:{risk}")
            runtime_risks.append(risk)
        if node.op_type == "Einsum" and any(types.get(name) == TensorProto.UINT8 for name in node.input):
            risk = "Einsum:uint8"
            forbidden_ops.append(f"node_{index}:{risk}")
            runtime_risks.append(risk)
        if node.op_type == "Where" and len(node.input) >= 3:
            branch_type = types.get(node.input[1]) or types.get(node.input[2])
            if branch_type in {TensorProto.INT8, TensorProto.INT16}:
                risk = f"Where:{TYPE_NAMES.get(branch_type, branch_type)}"
                forbidden_ops.append(f"node_{index}:{risk}")
                runtime_risks.append(risk)
        if node.op_type == "Sum" and len(node.input) > 2 and input_type in INTEGER_TYPES:
            risk = f"Sum:{input_type_name}:variadic"
            forbidden_ops.append(f"node_{index}:{risk}")
            runtime_risks.append(risk)
        if (
            node.op_type == "ScatterND"
            and len(node.input) >= 2
            and types.get(node.input[1]) == TensorProto.INT32
        ):
            risk = "ScatterND:int32"
            forbidden_ops.append(f"node_{index}:{risk}")
            runtime_risks.append(risk)
        if node.op_type in {"Conv", "QLinearConv"}:
            pads = next((attr for attr in node.attribute if attr.name == "pads"), None)
            if pads is not None and any(value < 0 for value in pads.ints):
                negative_padding.append(
                    {"node": node.name or f"node_{index}", "pads": list(pads.ints)}
                )
    return {
        "operators": dict(sorted(operators.items())),
        "forbidden_ops": forbidden_ops,
        "runtime_risks": sorted(runtime_risks),
        "negative_padding": negative_padding,
        "risk_fingerprint": {
            "runtime": sorted(runtime_risks),
            "negative_padding": sorted(
                f"{item['pads']}" for item in negative_padding
            ),
        },
        "runtime_compatible": not forbidden_ops and not negative_padding,
    }


def risk_delta(parent_audit: dict[str, Any], candidate_audit: dict[str, Any]) -> dict[str, list[str]]:
    def added(parent: list[str], candidate: list[str]) -> list[str]:
        remaining = Counter(parent)
        result: list[str] = []
        for value in candidate:
            if remaining[value] > 0:
                remaining[value] -= 1
            else:
                result.append(value)
        return sorted(result)

    parent_fingerprint = parent_audit.get("risk_fingerprint", {})
    candidate_fingerprint = candidate_audit.get("risk_fingerprint", {})
    return {
        "runtime": added(
            list(parent_fingerprint.get("runtime", [])),
            list(candidate_fingerprint.get("runtime", [])),
        ),
        "negative_padding": added(
            list(parent_fingerprint.get("negative_padding", [])),
            list(candidate_fingerprint.get("negative_padding", [])),
        ),
    }


def strict_record_errors(item: dict[str, Any]) -> list[str]:
    status = item.get("status")
    errors: list[str] = []
    if status not in ALL_STATES:
        return ["invalid_status"]
    if status not in PROGRESS_STATES:
        return errors

    checked = int(item.get("official_examples_checked", 0) or 0)
    passed = int(item.get("official_examples_passed", 0) or 0)
    if PROGRESS_STATES.index(status) >= PROGRESS_STATES.index("official_full_passed"):
        if checked <= 0 or checked != passed:
            errors.append("official_examples_incomplete")

    if PROGRESS_STATES.index(status) >= PROGRESS_STATES.index("fuzz_passed"):
        if status != "online_verified" and int(item.get("fuzz_trials", 0) or 0) < 2000:
            errors.append("insufficient_exact_fuzz")

    if status == "generator_passed":
        minimum = 50000 if item.get("risk_level") == "high" else 5000
        if int(item.get("generator_trials", 0) or 0) < minimum:
            errors.append("insufficient_generator_trials")

    if status in PROMOTABLE_STATES:
        if item.get("introduced_runtime_risks") or item.get("introduced_negative_padding"):
            errors.append("introduced_runtime_risk")
        if not item.get("runtime_compatible") and not item.get("inherited_parent_risk"):
            errors.append("runtime_not_safe")
        cost = item.get("cost")
        parent_cost = item.get("parent_cost")
        if cost is None or parent_cost is None or not float(cost) < float(parent_cost):
            errors.append("cost_not_lower_than_parent")

    if status == "canonical":
        models_passed = int(item.get("full400_models_passed", 0) or 0)
        full_checked = int(item.get("full400_examples_checked", 0) or 0)
        full_passed = int(item.get("full400_examples_passed", 0) or 0)
        if models_passed != 400 or full_checked <= 0 or full_checked != full_passed:
            errors.append("full400_evidence_missing")

    if status == "online_verified":
        if str(item.get("online_status", "")).upper() != "COMPLETE":
            errors.append("online_status_not_complete")
        if int(item.get("online_kaggle_ref", 0) or 0) <= 0:
            errors.append("online_ref_missing")
        if item.get("online_public_score") is None:
            errors.append("online_score_missing")
        if item.get("online_package_delta") is None:
            errors.append("online_delta_missing")
    return errors


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
    parent_audit = operator_audit(parent)
    introduced = risk_delta(parent_audit, audit)
    inherited_parent_risk = bool(
        (audit["runtime_risks"] or audit["negative_padding"])
        and not introduced["runtime"]
        and not introduced["negative_padding"]
    )
    if (introduced["runtime"] or introduced["negative_padding"]) and status in PROMOTABLE_STATES:
        raise RuntimeError("candidate introduces runtime risk and cannot be promotable")
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
    parent_cost = record.get("parent_cost")
    if parent_cost is None and record.get("cost") is not None and record.get("delta_cost") is not None:
        parent_cost = float(record["cost"]) + float(record["delta_cost"])
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
        "inherited_parent_risk": inherited_parent_risk,
        "introduced_runtime_risks": introduced["runtime"],
        "introduced_negative_padding": introduced["negative_padding"],
        "forbidden_ops": audit["forbidden_ops"],
        "negative_padding": audit["negative_padding"],
        "public_example_selected": bool(record.get("public_example_selected", False)),
        "cost": record.get("cost"),
        "parent_cost": parent_cost,
        "points": record.get("points"),
        "delta_cost": record.get("delta_cost"),
        "delta_points": record.get("delta_points"),
        "created_by": record.get("created_by", "candidate_registry.py"),
        "created_at": record.get("created_at", now),
        "updated_at": now,
        "notes": record.get("notes", ""),
        "operator_audit": audit,
        "parent_operator_audit": parent_audit,
        "proof_class": record.get("proof_class", ""),
        "checker_passed": bool(record.get("checker_passed", False)),
        "shape_inference_passed": bool(record.get("shape_inference_passed", False)),
        "full400_models_passed": int(record.get("full400_models_passed", 0)),
        "full400_examples_checked": int(record.get("full400_examples_checked", 0)),
        "full400_examples_passed": int(record.get("full400_examples_passed", 0)),
        "online_status": record.get("online_status", ""),
        "online_kaggle_ref": record.get("online_kaggle_ref"),
        "online_public_score": record.get("online_public_score"),
        "online_package_delta": record.get("online_package_delta"),
    }
    strict_errors = strict_record_errors(payload)
    if strict_errors and status in PROMOTABLE_STATES:
        raise RuntimeError(f"candidate evidence does not support {status}: {strict_errors}")
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
        if item.get("introduced_runtime_risks") or item.get("introduced_negative_padding"):
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
            if item.get("introduced_runtime_risks") or item.get("introduced_negative_padding"):
                failures.append({"task": item.get("task"), "error": "unsafe_promotable"})
        if args.strict:
            for error in strict_record_errors(item):
                failures.append({"task": item.get("task"), "error": error})
    print(json.dumps({"records": len(registry["candidates"]), "failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
