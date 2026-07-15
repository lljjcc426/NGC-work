from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


HERE = Path(__file__).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_dir", type=Path)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(HERE.parent))
    from c_score_common import score_onnx
    from full400_safety import (
        assert_complete_onnx_directory,
        atomic_write_json,
        load_baseline_manifest,
        model_hashes,
    )

    baseline = load_baseline_manifest(args.baseline_manifest)
    assert_complete_onnx_directory(args.onnx_dir)
    candidate_hashes = model_hashes(args.onnx_dir)
    changed = [
        task for task in sorted(candidate_hashes)
        if candidate_hashes[task] != baseline["models"][task]
    ]

    task_results = {}
    failures = []
    for task in changed:
        result = asdict(
            score_onnx(task, args.onnx_dir / f"{task}.onnx", validate_all=True)
        )
        task_results[task] = result
        if not (
            result["ok"]
            and result["valid_all_checked"]
            and result["examples_checked"] == result["examples_passed"]
        ):
            failures.append({"task": task, "error": result.get("error", "")})

    inherited_valid = int(baseline.get("models_valid", 0)) == 400
    inherited_examples = int(baseline.get("examples_checked", 0))
    full_validation = inherited_valid and inherited_examples > 0 and not failures
    summary = {
        "validation_mode": "incremental_parent_sha_plus_full_replacements",
        "baseline_kaggle_ref": baseline["kaggle_ref"],
        "baseline_package_sha256": baseline["package_sha256"],
        "baseline_model_set_sha256": baseline["model_set_sha256"],
        "model_count": 400,
        "unchanged_model_count": 400 - len(changed),
        "changed_tasks": changed,
        "changed_models_passed": len(changed) - len(failures),
        "models_passed": 400 if full_validation else 400 - len(failures),
        "models_failed": len(failures),
        "examples_checked": inherited_examples,
        "examples_passed": inherited_examples if full_validation else None,
        "full_validation": full_validation,
        "failures": failures,
        "changed_task_results": task_results,
        "candidate_model_hashes": candidate_hashes,
    }
    atomic_write_json(args.output, summary)
    print(json.dumps(summary, separators=(",", ":")))
    return 0 if full_validation else 1


if __name__ == "__main__":
    raise SystemExit(main())
