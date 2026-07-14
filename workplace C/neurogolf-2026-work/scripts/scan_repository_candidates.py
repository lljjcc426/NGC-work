from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path


HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
DEFAULT_PARENT = REPO / "workplace C" / "artifacts" / "GOLF_20260714_FULL400_ROUND3_REBASE_7379_19" / "onnx"
DEFAULT_PARENT_MANIFEST = DEFAULT_PARENT.parent / "package_manifest.json"
DEFAULT_OUTPUT = REPO / "workplace C" / "artifacts" / "full400_round4_repository_scan.json"


def _score(job: tuple[str, str, int]) -> dict:
    task, raw_path, max_examples = job
    sys.path.insert(0, str(HERE.parent))
    from c_score_common import score_onnx

    return asdict(score_onnx(task, Path(raw_path), validate_all=max_examples == 0, max_examples=max_examples))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find lower-cost fully valid ONNX models already present in the repository.")
    parser.add_argument("--parent-dir", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--parent-manifest", type=Path, default=DEFAULT_PARENT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--exclude-task", action="append", default=["task349"])
    parser.add_argument(
        "--scan-root",
        type=Path,
        action="append",
        default=[],
        help="Additional directory containing taskXXX ONNX candidates.",
    )
    parser.add_argument("--only-scan-root", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(HERE.parent))
    from candidate_registry import load_registry, operator_audit
    from full400_safety import TASKS, atomic_write_json, sha256_file

    parent_manifest = json.loads(args.parent_manifest.read_text(encoding="utf-8"))
    parent_costs = {
        task: int(row["winner_cost"])
        for task, row in parent_manifest.get("tasks", {}).items()
        if row.get("winner_cost") is not None
    }
    parent_hashes = {task: sha256_file(args.parent_dir / f"{task}.onnx") for task in TASKS}
    registry_path = HERE.parent.parent / "config" / "candidate_registry.json"
    blocked_hashes = {
        item["candidate_sha256"]
        for item in load_registry(registry_path)["candidates"]
        if item.get("status") in {"blocked", "local_only", "rejected"}
    }
    excluded = set(args.exclude_task)
    pattern = re.compile(r"task(\d{3})", re.IGNORECASE)
    unique: dict[tuple[str, str], Path] = {}
    rejected_audit: list[dict] = []
    scan_roots = [] if args.only_scan_root else [REPO / f"workplace {group}" for group in "ABCDEF"]
    scan_roots.extend(path.resolve() for path in args.scan_root)
    for root in scan_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.onnx"):
            match = pattern.search(str(path))
            if not match:
                continue
            task = f"task{match.group(1)}"
            if task in excluded:
                continue
            digest = sha256_file(path)
            if digest == parent_hashes.get(task) or digest in blocked_hashes:
                continue
            key = (task, digest)
            if key in unique:
                continue
            try:
                audit = operator_audit(path)
            except Exception as exc:
                rejected_audit.append({"task": task, "path": str(path), "sha256": digest, "reason": f"audit:{type(exc).__name__}:{exc}"})
                continue
            parent_audit = operator_audit(args.parent_dir / f"{task}.onnx")
            inherited_parent_risk = bool(
                not audit["runtime_compatible"]
                and audit.get("forbidden_ops") == parent_audit.get("forbidden_ops")
                and audit.get("negative_padding") == parent_audit.get("negative_padding")
            )
            if not audit["runtime_compatible"] and not inherited_parent_risk:
                rejected_audit.append({"task": task, "path": str(path), "sha256": digest, "reason": "runtime_unsafe", "audit": audit})
                continue
            unique[key] = path

    missing_parent_costs = sorted({task for task, _ in unique} - set(parent_costs))
    if missing_parent_costs:
        parent_jobs = [(task, str(args.parent_dir / f"{task}.onnx"), 1) for task in missing_parent_costs]
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(_score, job): job for job in parent_jobs}
            for future in as_completed(futures):
                task, _, _ = futures[future]
                row = future.result()
                if not row.get("ok") or row.get("cost") is None:
                    raise RuntimeError(f"could not measure current parent cost for {task}: {row.get('error', '')}")
                parent_costs[task] = int(row["cost"])

    screen_jobs = [(task, str(path), 1) for (task, _), path in unique.items()]
    screen_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(_score, job): job for job in screen_jobs}
        for completed, future in enumerate(as_completed(futures), start=1):
            task, raw_path, _ = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {"task": task, "path": raw_path, "ok": False, "cost": None, "error": f"worker:{type(exc).__name__}:{exc}"}
            if row.get("ok") and row.get("cost") is not None and row["cost"] < parent_costs[task]:
                screen_rows.append(row)
            if completed % 100 == 0 or completed == len(screen_jobs):
                print(json.dumps({"stage": "screen", "completed": completed, "total": len(screen_jobs), "lower": len(screen_rows)}), flush=True)

    full_jobs = [(row["task"], row["path"], 0) for row in screen_rows]
    full_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(_score, job): job for job in full_jobs}
        for completed, future in enumerate(as_completed(futures), start=1):
            task, raw_path, _ = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {"task": task, "path": raw_path, "ok": False, "cost": None, "error": f"worker:{type(exc).__name__}:{exc}"}
            row["parent_cost"] = parent_costs[task]
            row["delta_cost"] = parent_costs[task] - row["cost"] if row.get("cost") is not None else None
            row["accepted_official"] = bool(
                row.get("ok")
                and row.get("examples_checked") == row.get("examples_passed")
                and row.get("cost") is not None
                and row["cost"] < parent_costs[task]
            )
            full_rows.append(row)
            if completed % 25 == 0 or completed == len(full_jobs):
                print(json.dumps({"stage": "full", "completed": completed, "total": len(full_jobs), "accepted_official": sum(item["accepted_official"] for item in full_rows)}), flush=True)

    payload = {
        "parent_dir": str(args.parent_dir.resolve()),
        "unique_candidates": len(unique),
        "audit_rejected": rejected_audit,
        "screen_lower": len(screen_rows),
        "full_results": sorted(full_rows, key=lambda row: (not row["accepted_official"], -(row.get("delta_cost") or 0), row["task"])),
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({
        "unique_candidates": len(unique),
        "audit_rejected": len(rejected_audit),
        "screen_lower": len(screen_rows),
        "accepted_official": sum(row["accepted_official"] for row in full_rows),
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
