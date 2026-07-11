from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

from c_score_common import KAGGLEGOLF_ROOT, SCORE_DOCS
from c_task_model_common import TARGET_COST, normalize_task, read_status, upsert_status


COMPETITION = "neurogolf-2026"
DEFAULT_PARENT = (
    KAGGLEGOLF_ROOT
    / "submissions"
    / "candidates"
    / "GOLF_20260709_101_prvsiyan_7266_72_repro"
)
DEFAULT_PARENT_PUBLIC = 7266.72
DAILY_SOFT_CAP = 90
LOG_PATH = SCORE_DOCS / "C_AUTOSUBMIT_LOG.csv"
LOG_FIELDS = [
    "created_at",
    "task",
    "exp_id",
    "parent_path",
    "parent_public_score",
    "artifact_path",
    "old_cost",
    "new_cost",
    "expected_delta",
    "public_candidate_score",
    "observed_delta",
    "online_verified",
    "status",
    "notes",
]


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "pass"}


def append_log(row: dict) -> None:
    exists = LOG_PATH.exists() and LOG_PATH.stat().st_size > 0
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in LOG_FIELDS})


def submission_snapshot() -> tuple[int, bool, str]:
    proc = subprocess.run(
        ["kaggle", "competitions", "submissions", "-c", COMPETITION],
        cwd=KAGGLEGOLF_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Kaggle submission history failed: {(proc.stderr or proc.stdout)[:500]}")
    today = date.today().isoformat()
    lines = proc.stdout.splitlines()
    used = sum(today in line for line in lines)
    pending = any(today in line and "PENDING" in line.upper() for line in lines)
    return used, pending, proc.stdout


def queue_row(exp_id: str) -> dict[str, str]:
    path = KAGGLEGOLF_ROOT / "experiments" / "submission_queue.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("exp_id") == exp_id:
                return row
    return {}


def run(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        command,
        cwd=KAGGLEGOLF_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.stderr:
        print(proc.stderr.rstrip(), file=sys.stderr)
    return proc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument(
        "--parent",
        type=Path,
        default=Path(os.environ.get("NEUROGOLF_SUBMISSION_PARENT", str(DEFAULT_PARENT))),
    )
    parser.add_argument("--parent-public-score", type=float, default=DEFAULT_PARENT_PUBLIC)
    parser.add_argument("--allow-intermediate", action="store_true")
    parser.add_argument("--min-intermediate-gain", type=float, default=0.0)
    parser.add_argument("--daily-soft-cap", type=int, default=DAILY_SOFT_CAP)
    parser.add_argument("--poll-timeout", type=int, default=300)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    task = normalize_task(args.task)
    status = next((row for row in read_status() if normalize_task(row["task"]) == task), None)
    if status is None:
        raise SystemExit(f"status row missing for {task}; run c_refresh_20plus_status.py")
    if not truthy(status.get("rule_valid")) or not truthy(status.get("onnx_valid")):
        raise SystemExit(f"{task} is not fully rule/ONNX validated")

    artifact = (args.artifact or Path(status["artifact_path"])).resolve()
    if not artifact.exists():
        raise FileNotFoundError(artifact)
    parent_onnx = args.parent / "onnx" if (args.parent / "onnx").exists() else args.parent
    if len(list(parent_onnx.glob("task*.onnx"))) != 400:
        raise SystemExit(f"parent does not contain exactly 400 ONNX files: {parent_onnx}")

    old_cost = int(float(status["current_cost"]))
    new_cost = int(float(status["best_cost"]))
    old_points = 25.0 - math.log(max(1, old_cost))
    new_points = 25.0 - math.log(max(1, new_cost))
    expected_delta = new_points - old_points
    target_met = new_cost <= TARGET_COST
    if not target_met and expected_delta <= args.min_intermediate_gain:
        raise SystemExit(
            f"submission gate failed: target_met={target_met}, expected_delta={expected_delta:.6f}; "
            "requires positive local delta"
        )

    used, pending, _ = submission_snapshot()
    if used >= args.daily_soft_cap:
        raise SystemExit(f"daily soft cap reached: {used}/{args.daily_soft_cap}")
    if pending:
        raise SystemExit("a Kaggle submission is still PENDING; no new submission started")

    exp_id = f"GOLF_{datetime.now().strftime('%Y%m%d')}_C20_{task.upper()}_{datetime.now().strftime('%H%M%S')}"
    base_log = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task": task,
        "exp_id": exp_id,
        "parent_path": str(args.parent),
        "parent_public_score": args.parent_public_score,
        "artifact_path": str(artifact),
        "old_cost": old_cost,
        "new_cost": new_cost,
        "expected_delta": expected_delta,
    }
    if not args.execute:
        append_log({**base_log, "status": "dry_run_ready", "notes": f"used_today={used}"})
        print({**base_log, "used_today": used, "status": "dry_run_ready"})
        return

    build = run(
        [
            sys.executable,
            "scripts/13_single_task_override.py",
            "--exp-id",
            exp_id,
            "--base",
            str(args.parent),
            "--task",
            task,
            "--model",
            str(artifact),
            "--source-id",
            "SRC_ARC_DSL_GITHUB",
            "--direction-id",
            "DIR_20260610_001_simple_exact_batch_replacement",
            "--leaderboard-source-id",
            "SRC_DISCUSSION_AGENT_HARNESS_6580",
            "--paper-source-id",
            "SRC_ARC_PRIZE_2024_REPORT",
            "--open-repo-source-id",
            "SRC_ARC_DSL_GITHUB",
            "--historical-competition-source-id",
            "SRC_GOOGLE_CODE_GOLF_2025_CGI_WRITEUP",
            "--risk",
            "medium",
            "--validate",
            "--pack",
            "--build-notebook",
            "--record",
        ],
        timeout=1800,
    )
    if build.returncode != 0:
        append_log({**base_log, "status": "build_failed", "notes": (build.stderr or build.stdout)[-500:]})
        raise SystemExit(build.returncode)

    submit = run(
        [
            sys.executable,
            "scripts/19_submit_queue.py",
            "--exp-id",
            exp_id,
            "--poll-after-submit",
            "--poll-timeout",
            str(args.poll_timeout),
        ],
        timeout=max(1200, args.poll_timeout + 900),
    )
    result = queue_row(exp_id)
    public_text = result.get("public_score", "")
    try:
        public_score = float(public_text)
    except (TypeError, ValueError):
        public_score = None
    observed_delta = public_score - args.parent_public_score if public_score is not None else None
    tolerance = max(0.05, abs(expected_delta) * 0.10)
    online_verified = bool(
        submit.returncode == 0
        and observed_delta is not None
        and observed_delta > 0
        and abs(observed_delta - expected_delta) <= tolerance
    )
    submission_status = result.get("status", "submit_command_failed" if submit.returncode else "unknown")
    append_log(
        {
            **base_log,
            "public_candidate_score": public_score if public_score is not None else "",
            "observed_delta": observed_delta if observed_delta is not None else "",
            "online_verified": str(online_verified).lower(),
            "status": submission_status,
            "notes": f"submit_returncode={submit.returncode}; tolerance={tolerance:.6f}",
        }
    )
    upsert_status(
        {
            "task": task,
            "public_parent_score": args.parent_public_score,
            "public_candidate_score": public_score if public_score is not None else "",
            "public_delta": observed_delta if observed_delta is not None else "",
            "online_verified": str(online_verified).lower(),
        }
    )
    if submit.returncode != 0 or public_score is None:
        raise SystemExit(submit.returncode or 4)
    print(
        {
            "task": task,
            "exp_id": exp_id,
            "public_score": public_score,
            "observed_delta": observed_delta,
            "expected_delta": expected_delta,
            "online_verified": online_verified,
        }
    )


if __name__ == "__main__":
    main()
