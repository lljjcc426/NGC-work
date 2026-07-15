from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from score_task_candidate_pool_robust import run_one


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["rows"] if isinstance(data, dict) else data


def save(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-dir", type=Path, required=True)
    parser.add_argument("--old-scores", type=Path, required=True)
    parser.add_argument("--new-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    old_rows = {int(row["task"]): row for row in load_rows(args.old_scores)}
    cached: dict[int, dict] = {}
    if args.resume and args.out.exists():
        cached = {int(row["task"]): row for row in load_rows(args.out)}

    rows: dict[int, dict] = {}
    jobs: list[tuple[int, str, str, int]] = []
    inherited = 0
    for task in range(1, 401):
        name = f"task{task:03d}.onnx"
        old_path = args.old_dir / name
        new_path = args.new_dir / name
        if not new_path.exists():
            raise SystemExit(f"missing {new_path}")
        if old_path.exists() and digest(old_path) == digest(new_path):
            row = dict(old_rows[task])
            row.update(label=args.new_dir.name, path=str(new_path), bytes=new_path.stat().st_size)
            rows[task] = row
            inherited += 1
        elif task in cached and cached[task].get("path") == str(new_path):
            rows[task] = cached[task]
        else:
            jobs.append((task, args.new_dir.name, str(new_path), args.timeout))

    print(f"inherited={inherited} cached={len(rows) - inherited} pending={len(jobs)}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one, task, label, path, timeout): task
            for task, label, path, timeout in jobs
        }
        for index, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            rows[task] = future.result()
            row = rows[task]
            print(
                f"[{index}/{len(jobs)}] task{task:03d} valid={row.get('valid')} "
                f"cost={row.get('cost')} error={row.get('error', '')}",
                flush=True,
            )
            save(args.out, [rows[key] for key in sorted(rows)])

    ordered = [rows[task] for task in range(1, 401)]
    save(args.out, ordered)
    valid = [row for row in ordered if row.get("valid")]
    total = sum(float(row["points"]) for row in valid)
    invalid = [int(row["task"]) for row in ordered if not row.get("valid")]
    print(f"done valid={len(valid)}/400 total={total:.12f} invalid={invalid}", flush=True)


if __name__ == "__main__":
    main()
