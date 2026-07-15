from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path


HERE = Path(__file__).resolve()


def _score(job: tuple[str, str]) -> dict:
    task, raw_path = job
    sys.path.insert(0, str(HERE.parent))
    from c_score_common import score_onnx

    return asdict(score_onnx(task, Path(raw_path), validate_all=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_dir", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    tasks = [f"task{index:03d}" for index in range(1, 401)]
    missing = [task for task in tasks if not (args.onnx_dir / f"{task}.onnx").is_file()]
    extra = sorted(
        path.name
        for path in args.onnx_dir.glob("task*.onnx")
        if path.stem not in set(tasks)
    )
    if missing or extra:
        print(json.dumps({"missing": missing, "extra": extra}, separators=(",", ":")))
        raise SystemExit(1)

    failures: list[dict] = []
    checked_examples = 0
    passed_examples = 0
    results: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(_score, (task, str(args.onnx_dir / f"{task}.onnx"))): task
            for task in tasks
        }
        completed = 0
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"task": task, "ok": False, "error": f"worker:{type(exc).__name__}:{exc}"}
            checked_examples += int(result.get("examples_checked") or 0)
            passed_examples += int(result.get("examples_passed") or 0)
            results[task] = result
            if not (
                result.get("ok")
                and result.get("valid_all_checked")
                and result.get("examples_checked") == result.get("examples_passed")
            ):
                failures.append(result)
            completed += 1
            if completed % 50 == 0:
                print(json.dumps({"completed": completed, "total": 400}), flush=True)

    summary = {
        "model_count": 400,
        "models_passed": 400 - len(failures),
        "models_failed": len(failures),
        "examples_checked": checked_examples,
        "examples_passed": passed_examples,
        "full_validation": not failures,
        "failures": [
            {"task": item.get("task"), "error": item.get("error", "")}
            for item in failures
        ],
        "tasks": {task: results[task] for task in sorted(results)},
    }
    if args.output:
        sys.path.insert(0, str(HERE.parent))
        from full400_safety import atomic_write_json

        atomic_write_json(args.output, summary)
    print(json.dumps(summary, separators=(",", ":")))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
