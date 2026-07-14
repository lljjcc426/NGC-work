from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve()
PROJECT = HERE.parent.parent
REPO = HERE.parents[3]
DEFAULT_BASELINE = PROJECT / "config" / "baseline_manifest.json"
TASK_PATTERN = re.compile(r"task(\d{3})")


def _score(job: tuple[str, str, int]) -> dict[str, Any]:
    task, raw_path, max_examples = job
    sys.path.insert(0, str(HERE.parent))
    from c_score_common import score_onnx

    return asdict(
        score_onnx(
            task,
            Path(raw_path),
            validate_all=max_examples == 0,
            max_examples=max_examples,
        )
    )


def _task(path: Path) -> str | None:
    match = TASK_PATTERN.search(str(path))
    if match is None:
        return None
    value = int(match.group(1))
    return f"task{value:03d}" if 1 <= value <= 400 else None


def _candidate_paths(
    roots: list[Path], *, include_debug: bool = False
) -> list[tuple[str, Path]]:
    results: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.onnx"):
            resolved = path.resolve()
            lowered_parts = {part.lower() for part in path.parts}
            if resolved in seen or (not include_debug and "debug" in lowered_parts):
                continue
            if (
                "onnx" not in lowered_parts
                and "optimized_onnx" not in lowered_parts
                and not (include_debug and "debug" in lowered_parts)
            ):
                continue
            task = _task(path)
            if task is not None:
                results.append((task, resolved))
                seen.add(resolved)
    return sorted(results, key=lambda item: (item[0], str(item[1])))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Screen canonical local task candidates against the current parent."
    )
    parser.add_argument("--baseline-manifest", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-examples", type=int, default=1)
    parser.add_argument("--root", type=Path, action="append")
    parser.add_argument("--include-debug", action="store_true")
    args = parser.parse_args()

    baseline = json.loads(args.baseline_manifest.read_text(encoding="utf-8"))
    parent_dir = Path(baseline["onnx_dir"])
    roots = args.root or [
        REPO / "workplace A",
        REPO / "workplace B",
        REPO / "workplace C" / "single_task",
        REPO / "workplace D",
        REPO / "workplace E",
    ]
    candidates = _candidate_paths(roots, include_debug=args.include_debug)
    tasks = sorted({task for task, _ in candidates})
    jobs = [(task, str(parent_dir / f"{task}.onnx"), args.max_examples) for task in tasks]
    jobs.extend((task, str(path), args.max_examples) for task, path in candidates)

    scored: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_score, job): job for job in jobs}
        for future in as_completed(futures):
            task, path, _ = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                record = {"task": task, "path": path, "ok": False, "error": str(exc)}
            scored.append(record)

    parent_paths = {(parent_dir / f"{task}.onnx").resolve() for task in tasks}
    parents = {
        record["task"]: record
        for record in scored
        if Path(record["path"]).resolve() in parent_paths
    }
    lower = []
    for record in scored:
        path = Path(record["path"]).resolve()
        if path in parent_paths or not record.get("ok"):
            continue
        parent = parents.get(record["task"])
        if parent is None or not parent.get("ok"):
            continue
        if record.get("examples_passed") != record.get("examples_checked"):
            continue
        if int(record["cost"]) < int(parent["cost"]):
            lower.append(
                {
                    "task": record["task"],
                    "candidate_path": record["path"],
                    "candidate_cost": int(record["cost"]),
                    "parent_cost": int(parent["cost"]),
                    "delta_cost": int(parent["cost"]) - int(record["cost"]),
                    "candidate_sha256": record["sha256"],
                }
            )
    lower.sort(key=lambda item: (-item["delta_cost"], item["task"], item["candidate_path"]))
    payload = {
        "baseline": baseline,
        "candidate_count": len(candidates),
        "task_count": len(tasks),
        "lower_cost_candidates": lower,
        "score_failures": [record for record in scored if not record.get("ok")],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"candidates": len(candidates), "tasks": len(tasks), "lower": len(lower)}))
    for item in lower:
        print(json.dumps(item, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
