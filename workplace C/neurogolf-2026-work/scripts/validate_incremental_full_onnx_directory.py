from __future__ import annotations

import argparse
import json
from pathlib import Path

from c_score_common import score_onnx, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reuse full validation for byte-identical ONNX files and revalidate changed tasks."
    )
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--reference-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected = [f"task{i:03d}" for i in range(1, 401)]
    candidate_files = sorted(args.candidate_dir.glob("task*.onnx"))
    if [p.stem for p in candidate_files] != expected:
        raise RuntimeError("candidate directory must contain exactly task001-task400")

    reference = json.loads(args.reference_validation.read_text(encoding="utf-8"))
    if not reference.get("full_validation") or reference.get("models_passed") != 400:
        raise RuntimeError("reference validation is not a complete 400/400 pass")
    reference_tasks = reference.get("tasks", {})
    if sorted(reference_tasks) != expected:
        raise RuntimeError("reference validation does not contain exactly 400 task records")

    tasks: dict[str, dict] = {}
    changed: list[str] = []
    reused: list[str] = []
    failures: list[dict] = []
    for task in expected:
        candidate_path = args.candidate_dir / f"{task}.onnx"
        reference_path = args.reference_dir / f"{task}.onnx"
        candidate_sha = sha256_file(candidate_path)
        reference_sha = sha256_file(reference_path)
        record = reference_tasks[task]
        if reference_sha != record.get("sha256"):
            raise RuntimeError(f"reference SHA mismatch for {task}")
        if candidate_sha == reference_sha:
            item = dict(record)
            item["path"] = str(candidate_path)
            item["validation_source"] = "reused_byte_identical_reference"
            tasks[task] = item
            reused.append(task)
            continue

        changed.append(task)
        result = score_onnx(task, candidate_path)
        item = dict(result.__dict__)
        item["validation_source"] = "fresh_official_full"
        tasks[task] = item
        if not (
            result.ok
            and result.valid_all_checked
            and result.examples_checked == result.examples_passed
        ):
            failures.append({"task": task, "error": result.error})

    output = {
        "model_count": 400,
        "models_passed": sum(
            1
            for item in tasks.values()
            if item.get("ok") and item.get("examples_checked") == item.get("examples_passed")
        ),
        "models_failed": len(failures),
        "examples_checked": sum(int(item.get("examples_checked") or 0) for item in tasks.values()),
        "examples_passed": sum(int(item.get("examples_passed") or 0) for item in tasks.values()),
        "full_validation": not failures and len(tasks) == 400,
        "validation_mode": "sha256_incremental",
        "reference_validation": str(args.reference_validation),
        "reused_task_count": len(reused),
        "fresh_task_count": len(changed),
        "fresh_tasks": changed,
        "failures": failures,
        "tasks": tasks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "tasks"}, indent=2))
    return 0 if output["full_validation"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
