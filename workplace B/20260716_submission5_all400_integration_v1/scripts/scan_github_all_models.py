from __future__ import annotations

import hashlib
import json
import re
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "external_repos" / "NGC-work"
BASE = ROOT / "team_baselines" / "team_submission5_20260716" / "extracted"
WORK = (
    ROOT
    / "public_probe_variants"
    / "team_submission5_b_work_20260716"
    / "submission"
)
OUT = ROOT / "artifacts" / "github_all_candidate_manifest_20260716.json"
POOL = ROOT / "github_all_candidates_20260716"
TASK_RE = re.compile(r"task(\d{3})", re.IGNORECASE)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def task_from_name(name: str) -> int | None:
    match = TASK_RE.search(Path(name).name)
    if not match:
        return None
    task = int(match.group(1))
    return task if 1 <= task <= 400 else None


def main() -> None:
    result = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    files = [item.decode("utf-8") for item in result.stdout.split(b"\0") if item]
    onnx_files = [name for name in files if name.lower().endswith(".onnx")]
    zip_files = [name for name in files if name.lower().endswith(".zip")]

    records: dict[tuple[int, str], dict] = {}
    payloads: dict[tuple[int, str], bytes] = {}

    def add(task: int, data: bytes, source: str) -> None:
        sha = digest(data)
        key = (task, sha)
        if key not in records:
            records[key] = {
                "task": task,
                "sha256": sha,
                "bytes": len(data),
                "sources": [],
            }
            payloads[key] = data
        records[key]["sources"].append(source)

    for relative in onnx_files:
        task = task_from_name(relative)
        if task is not None:
            add(task, (REPO / relative).read_bytes(), f"file:{relative}")

    for relative in zip_files:
        with zipfile.ZipFile(REPO / relative) as archive:
            for info in archive.infolist():
                task = task_from_name(info.filename)
                if task is None or not info.filename.lower().endswith(".onnx"):
                    continue
                add(task, archive.read(info), f"zip:{relative}!{info.filename}")

    base_hashes = {
        task: digest((BASE / f"task{task:03d}.onnx").read_bytes())
        for task in range(1, 401)
    }
    work_hashes = {
        task: digest((WORK / f"task{task:03d}.onnx").read_bytes())
        for task in range(1, 401)
    }
    per_task = defaultdict(int)
    rows = []
    for key, record in sorted(records.items()):
        task, sha = key
        record["matches_submission5"] = sha == base_hashes[task]
        record["matches_working"] = sha == work_hashes[task]
        if not record["matches_working"]:
            destination = POOL / f"task{task:03d}" / sha[:16] / f"task{task:03d}.onnx"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists() or digest(destination.read_bytes()) != sha:
                destination.write_bytes(payloads[key])
            record["materialized"] = str(destination)
        record["sources"].sort()
        rows.append(record)
        per_task[task] += 1

    report = {
        "repo": str(REPO),
        "tracked_onnx_files": len(onnx_files),
        "tracked_zip_files": len(zip_files),
        "unique_models": len(rows),
        "unique_not_working": sum(not row["matches_working"] for row in rows),
        "tasks_with_candidates": len(per_task),
        "tasks_with_multiple_models": sum(count > 1 for count in per_task.values()),
        "per_task_unique": dict(sorted(per_task.items())),
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    print(f"manifest={OUT}")


if __name__ == "__main__":
    main()
