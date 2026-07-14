from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import onnx


HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
DEFAULT_PARENT = Path(
    r"E:/kagglegolf/submissions/candidates/"
    r"GOLF_20260713_SUBMISSION8_REBASE/onnx"
)
DEFAULT_WORK = REPO / "workplace C" / "artifacts" / "full400_exact_cleanup"
ASSIGNMENTS = REPO / "assignments" / "task_assignment_400.csv"
SAFE_TRANSFORMS = [
    "pad_axes",
    "constant_pad_axes",
    "broadcast_init",
    "trivial_nodes",
    "optional_defaults",
    "neutral_elementwise",
    "unit_reduction_axes",
    "constant_dedup",
    "init_cleanup",
]


def _load_sweep_module():
    path = HERE.parent / "cd_parent_micro_sweep.py"
    spec = importlib.util.spec_from_file_location("ngc_exact_sweep", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _score_job(job: tuple[str, str, str, str]) -> dict:
    task, parent_raw, candidate_raw, canonical_raw = job
    sys.path.insert(0, str(HERE.parent))
    from c_score_common import score_onnx

    parent = score_onnx(task, Path(parent_raw), validate_all=True)
    candidate = score_onnx(task, Path(candidate_raw), validate_all=True)
    canonical = None
    canonical_path = Path(canonical_raw) if canonical_raw else None
    if canonical_path is not None and canonical_path.is_file():
        canonical = score_onnx(task, canonical_path, validate_all=True)
    return {
        "task": task,
        "parent": asdict(parent),
        "candidate": asdict(candidate),
        "canonical": asdict(canonical) if canonical is not None else None,
    }


def _owners() -> dict[str, str]:
    with ASSIGNMENTS.open(newline="", encoding="utf-8-sig") as handle:
        return {row["task"]: row["owner"] for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-dir", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tasks", default="")
    parser.add_argument(
        "--source-mode",
        choices=("parent", "canonical"),
        default="parent",
        help="Apply transforms to the parent or the current canonical when present.",
    )
    parser.add_argument(
        "--transforms",
        default=",".join(SAFE_TRANSFORMS),
        help="Comma-separated transform names to apply in order.",
    )
    args = parser.parse_args()

    owners = _owners()
    requested = {
        item.strip() for item in args.tasks.split(",") if item.strip()
    }
    tasks = sorted(requested or {f"task{index:03d}" for index in range(1, 401)})
    transforms = [
        item.strip() for item in args.transforms.split(",") if item.strip()
    ]
    if not transforms:
        parser.error("--transforms must contain at least one transform name")
    sweep = _load_sweep_module()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, str, str, str]] = []
    changes_by_task: dict[str, list[str]] = {}
    canonical_by_task: dict[str, Path] = {}
    for task in tasks:
        parent = args.parent_dir / f"{task}.onnx"
        owner = owners[task]
        canonical = (
            REPO
            / f"workplace {owner}"
            / "single_task"
            / task
            / "onnx"
            / f"{task}_candidate.onnx"
        )
        source_path = (
            canonical
            if args.source_mode == "canonical" and canonical.is_file()
            else parent
        )
        source = onnx.load(source_path)
        try:
            candidate, changes = sweep.apply_transforms(source, transforms)
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "task": task,
                        "stage": "transform",
                        "rejected": f"{type(exc).__name__}:{exc}",
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            continue
        if not changes:
            continue
        candidate.producer_name = "ngc_full400_exact_cleanup"
        candidate_path = args.work_dir / f"{task}.onnx"
        try:
            onnx.checker.check_model(candidate, full_check=True)
            candidate = onnx.shape_inference.infer_shapes(candidate, strict_mode=True)
            onnx.checker.check_model(candidate, full_check=True)
            onnx.save(candidate, candidate_path)
        except Exception as exc:
            candidate_path.unlink(missing_ok=True)
            print(
                json.dumps(
                    {
                        "task": task,
                        "stage": "checker",
                        "changes": changes,
                        "rejected": f"{type(exc).__name__}:{exc}",
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            continue

        changes_by_task[task] = changes
        canonical_by_task[task] = canonical
        jobs.append((task, str(parent), str(candidate_path), str(canonical)))

    promoted = 0
    total_delta_cost = 0
    total_delta_points = 0.0
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(_score_job, job): job[0] for job in jobs}
        for future in as_completed(futures):
            task = futures[future]
            result = future.result()
            parent = result["parent"]
            candidate = result["candidate"]
            canonical = result["canonical"]
            candidate_path = args.work_dir / f"{task}.onnx"
            canonical_path = canonical_by_task[task]

            accepted = bool(
                parent["ok"]
                and candidate["ok"]
                and candidate["examples_checked"] == candidate["examples_passed"]
                and candidate["cost"] is not None
                and parent["cost"] is not None
                and candidate["cost"] < parent["cost"]
            )
            canonical_better = bool(
                canonical
                and canonical["ok"]
                and canonical["cost"] is not None
                and candidate["cost"] is not None
                and canonical["cost"] <= candidate["cost"]
            )
            did_promote = accepted and not canonical_better
            if did_promote:
                canonical_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate_path, canonical_path)
                promoted += 1
                delta_cost = int(parent["cost"] - candidate["cost"])
                delta_points = float(candidate["points"] - parent["points"])
                total_delta_cost += delta_cost
                total_delta_points += delta_points
            else:
                candidate_path.unlink(missing_ok=True)
                delta_cost = 0
                delta_points = 0.0

            print(
                json.dumps(
                    {
                        "task": task,
                        "changes": changes_by_task[task],
                        "parent_cost": parent["cost"],
                        "candidate_cost": candidate["cost"],
                        "canonical_cost": canonical["cost"] if canonical else None,
                        "full_validation": (
                            f"{candidate['examples_passed']}/"
                            f"{candidate['examples_checked']}"
                        ),
                        "promoted": did_promote,
                        "delta_cost": delta_cost,
                        "delta_points": delta_points,
                        "error": candidate["error"],
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )

    print(
        json.dumps(
            {
                "tasks_scanned": len(tasks),
                "transforms": transforms,
                "opportunity_tasks": len(jobs),
                "promoted": promoted,
                "total_delta_cost": total_delta_cost,
                "total_delta_points": total_delta_points,
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
