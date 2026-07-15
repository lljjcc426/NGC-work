from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

sys.path.insert(0, str(ROOT / "src"))
import build_blend  # noqa: E402
from score_task_candidate_pool import collect_candidates, parse_tasks, save  # noqa: E402


def score_one(task: int, label: str, path: Path) -> None:
    row = build_blend.validate_and_score((task, label, str(path)))
    print(json.dumps(row, ensure_ascii=False), flush=True)


def run_one(task: int, label: str, path: str, timeout: int) -> dict:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--one",
        "--task",
        str(task),
        "--label",
        label,
        "--path",
        path,
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "task": task,
            "label": label,
            "path": path,
            "valid": False,
            "error": f"timeout>{timeout}s",
        }
    except Exception as exc:
        return {
            "task": task,
            "label": label,
            "path": path,
            "valid": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    if proc.returncode != 0:
        return {
            "task": task,
            "label": label,
            "path": path,
            "valid": False,
            "error": (proc.stderr or proc.stdout).strip()[-1200:] or f"returncode={proc.returncode}",
        }
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return {
            "task": task,
            "label": label,
            "path": path,
            "valid": False,
            "error": "no stdout",
        }
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        return {
            "task": task,
            "label": label,
            "path": path,
            "valid": False,
            "error": f"json decode: {exc}: {lines[-1][-300:]}",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    batch_required = "--one" not in sys.argv
    parser.add_argument("--base-name", required=batch_required)
    parser.add_argument("--tasks", required=batch_required)
    parser.add_argument("--roots", required=batch_required)
    parser.add_argument("--max-per-task", type=int, default=80)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--out", type=Path, required=batch_required)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--one", action="store_true")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--label", default="")
    parser.add_argument("--path", type=Path)
    args = parser.parse_args()

    if args.one:
        if args.path is None:
            raise SystemExit("--path required with --one")
        score_one(args.task, args.label, args.path)
        return

    tasks = parse_tasks(args.tasks)
    roots = [ROOT / item for item in args.roots.replace(",", " ").split() if item.strip()]
    jobs = collect_candidates(args.base_name, tasks, roots, args.max_per_task)

    base_rows = json.loads(
        (ARTIFACTS / f"{args.base_name}_all_scores.json").read_text(encoding="utf-8")
    )
    base_by_task = {int(row["task"]): row for row in base_rows if row.get("valid")}

    cached: dict[str, dict] = {}
    rows: list[dict] = []
    if args.resume and args.out.exists():
        for row in json.loads(args.out.read_text(encoding="utf-8")):
            if row.get("path"):
                cached[row["path"]] = row
        rows.extend(cached.values())
        print(f"resume cached={len(cached)}", flush=True)

    pending = [job for job in jobs if job[2] not in cached]
    print(f"scoring pending={len(pending)} total={len(jobs)}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_job = {
            pool.submit(run_one, task, label, path, args.timeout): (task, label, path)
            for task, label, path in pending
        }
        for index, future in enumerate(as_completed(future_to_job), 1):
            task, label, path = future_to_job[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "task": task,
                    "label": label,
                    "path": path,
                    "valid": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            base = base_by_task.get(int(row["task"]))
            if base and row.get("valid") and row.get("points") is not None:
                row["base_points"] = base["points"]
                row["base_cost"] = base["cost"]
                row["gain"] = float(row["points"]) - float(base["points"])
            else:
                row["gain"] = None
            rows.append(row)
            if row.get("gain") is not None and row["gain"] > 0:
                print(
                    f"positive task{int(row['task']):03d} gain={row['gain']:+.6f} "
                    f"cost {row.get('base_cost')}->{row.get('cost')} {row['path']}",
                    flush=True,
                )
            if index % 10 == 0:
                save(args.out, rows)
                best = sorted(
                    [r for r in rows if r.get("gain") is not None],
                    key=lambda r: r["gain"],
                    reverse=True,
                )[:5]
                print(
                    f"scored={index}/{len(pending)} best="
                    f"{[(r['task'], round(r['gain'], 4)) for r in best]}",
                    flush=True,
                )

    rows.sort(
        key=lambda row: row["gain"] if row.get("gain") is not None else -999.0,
        reverse=True,
    )
    save(args.out, rows)
    positives = [row for row in rows if row.get("gain") is not None and row["gain"] > 0]
    save(args.out.with_name(args.out.stem + "_positives.json"), positives)
    print(f"done positives={len(positives)} out={args.out}", flush=True)
    for row in positives[:40]:
        print(
            f"  task{int(row['task']):03d} gain={row['gain']:+.6f} "
            f"cost {row['base_cost']}->{row['cost']} {row['path']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
