from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path


HERE = Path(__file__).resolve()


def score_job(job: tuple[str, str]) -> dict:
    task, path = job
    sys.path.insert(0, str(HERE.parent))
    from c_score_common import score_onnx

    return asdict(score_onnx(task, Path(path), validate_all=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("external_dir", type=Path)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    manifest = json.loads(args.parent_manifest.read_text(encoding="utf-8"))
    parent_tasks = manifest.get("tasks", {})
    files = sorted(args.external_dir.rglob("task*.onnx"))
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(score_job, (path.stem, str(path.resolve()))): path for path in files
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {"task": path.stem, "path": str(path), "ok": False, "error": str(exc)}
            parent = parent_tasks.get(path.stem, {})
            parent_cost = parent.get("winner_cost")
            row["parent_cost"] = parent_cost
            row["delta_points"] = (
                math.log(parent_cost / row["cost"])
                if row.get("ok") and parent_cost and row.get("cost") and row["cost"] < parent_cost
                else 0.0
            )
            rows.append(row)
    rows.sort(key=lambda row: (-float(row.get("delta_points") or 0), row["task"]))
    print(json.dumps({
        "files": len(files),
        "valid": sum(bool(row.get("ok")) for row in rows),
        "lower_than_parent": [row for row in rows if row.get("delta_points", 0) > 0],
        "invalid": [
            {"task": row["task"], "error": row.get("error", "")}
            for row in rows if not row.get("ok")
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
