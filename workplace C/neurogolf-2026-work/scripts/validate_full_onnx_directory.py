from __future__ import annotations

import argparse
import hashlib
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_path(output: Path | None, onnx_dir: Path) -> Path:
    if output:
        return output.with_suffix(output.suffix + ".checkpoint")
    return onnx_dir / ".full_validation.checkpoint.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_dir", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--chunk-size", type=int, default=40)
    parser.add_argument("--resume", action="store_true")
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

    results: dict[str, dict] = {}
    checkpoint = _checkpoint_path(args.output, args.onnx_dir)
    if args.resume and checkpoint.exists():
        try:
            cached = json.loads(checkpoint.read_text(encoding="utf-8"))
            for task, result in cached.get("tasks", {}).items():
                model = args.onnx_dir / f"{task}.onnx"
                if model.is_file() and result.get("sha256") == _sha256(model):
                    results[task] = result
        except Exception:
            results = {}

    pending = [task for task in tasks if task not in results]
    chunk_size = max(1, args.chunk_size)
    for offset in range(0, len(pending), chunk_size):
        chunk = pending[offset : offset + chunk_size]
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(_score, (task, str(args.onnx_dir / f"{task}.onnx"))): task
                for task in chunk
            }
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "task": task,
                        "ok": False,
                        "error": f"worker:{type(exc).__name__}:{exc}",
                    }
                results[task] = result
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temp = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
        temp.write_text(json.dumps({"tasks": results}, separators=(",", ":")), encoding="utf-8")
        temp.replace(checkpoint)
        print(json.dumps({"completed": len(results), "total": 400}), flush=True)

    failures: list[dict] = []
    checked_examples = 0
    passed_examples = 0
    for task in tasks:
        result = results[task]
        checked_examples += int(result.get("examples_checked") or 0)
        passed_examples += int(result.get("examples_passed") or 0)
        if not (
            result.get("ok")
            and result.get("valid_all_checked")
            and result.get("examples_checked") == result.get("examples_passed")
        ):
            failures.append(result)

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
    checkpoint.unlink(missing_ok=True)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
