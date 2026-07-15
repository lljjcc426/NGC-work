from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path


HERE = Path(__file__).resolve().parent


def _score(job: tuple[str, str, int]) -> dict:
    task, raw_path, max_examples = job
    sys.path.insert(0, str(HERE))
    from c_score_common import score_onnx

    return asdict(
        score_onnx(
            task,
            Path(raw_path),
            validate_all=max_examples == 0,
            max_examples=max_examples,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen a partial ONNX candidate directory against parent validation costs."
    )
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--parent-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-examples", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    parent = json.loads(args.parent_validation.read_text(encoding="utf-8"))["tasks"]
    jobs = []
    for path in sorted(args.candidate_dir.glob("task[0-9][0-9][0-9].onnx")):
        jobs.append((path.stem, str(path.resolve()), args.max_examples))

    rows = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(_score, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), start=1):
            task, raw_path, _ = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "task": task,
                    "path": raw_path,
                    "ok": False,
                    "cost": None,
                    "error": f"worker:{type(exc).__name__}:{exc}",
                }
            baseline = parent.get(task, {})
            result["parent_cost"] = baseline.get("cost")
            result["parent_points"] = baseline.get("points")
            result["improved"] = bool(
                result.get("ok")
                and result.get("cost") is not None
                and baseline.get("cost") is not None
                and result["cost"] < baseline["cost"]
            )
            if result["improved"]:
                result["delta_cost"] = baseline["cost"] - result["cost"]
                result["delta_points"] = result["points"] - baseline["points"]
            rows.append(result)
            if index % max(1, args.progress_every) == 0 or index == len(jobs):
                print(
                    json.dumps(
                        {
                            "completed": index,
                            "total": len(jobs),
                            "task": task,
                            "ok": result.get("ok"),
                            "improved": result.get("improved"),
                            "error": result.get("error"),
                        }
                    ),
                    flush=True,
                )

    rows.sort(key=lambda row: row["task"])
    payload = {
        "candidate_dir": str(args.candidate_dir.resolve()),
        "parent_validation": str(args.parent_validation.resolve()),
        "max_examples": args.max_examples,
        "tasks_scored": len(rows),
        "improved": [row for row in rows if row.get("improved")],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "tasks_scored": len(rows),
        "improved": len(payload["improved"]),
        "output": str(args.output.resolve()),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
