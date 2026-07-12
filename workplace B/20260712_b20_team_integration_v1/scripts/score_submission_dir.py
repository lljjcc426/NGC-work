from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import build_blend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    jobs = [
        (task, args.submission.name, str(args.submission / f"task{task:03d}.onnx"))
        for task in range(1, 401)
    ]
    missing = [path for _, _, path in jobs if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} task files; first={missing[0]}")

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(build_blend.validate_and_score, job): job for job in jobs
        }
        for index, future in enumerate(as_completed(futures), 1):
            job = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "task": job[0],
                    "label": job[1],
                    "path": job[2],
                    "valid": False,
                    "points": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            rows.append(row)
            if index % 25 == 0:
                print(f"scored={index}/400", flush=True)

    rows.sort(key=lambda row: int(row["task"]))
    invalid = [row for row in rows if not row.get("valid")]
    total = sum(float(row["points"]) for row in rows if row.get("valid"))
    report = {
        "submission": str(args.submission),
        "valid_tasks": len(rows) - len(invalid),
        "invalid_tasks": len(invalid),
        "total_points": total if not invalid else None,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "valid_tasks": report["valid_tasks"],
                "invalid_tasks": report["invalid_tasks"],
                "total_points": report["total_points"],
                "out": str(args.out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
