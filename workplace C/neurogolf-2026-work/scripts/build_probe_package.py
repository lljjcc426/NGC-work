from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import asdict
from pathlib import Path


HERE = Path(__file__).resolve()
PROJECT = HERE.parent.parent
DEFAULT_BASELINE = PROJECT / "config" / "baseline_manifest.json"


def parse_replacement(value: str) -> tuple[str, Path]:
    task, separator, raw_path = value.partition("=")
    if not separator or not task.startswith("task"):
        raise argparse.ArgumentTypeError("replacement must be taskXXX=path.onnx")
    path = Path(raw_path)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"candidate does not exist: {path}")
    return task, path


def parse_online_exception(value: str) -> tuple[str, tuple[str, str]]:
    task, separator, evidence = value.partition("=")
    digest, at, kaggle_ref = evidence.partition("@")
    if (
        not separator
        or not at
        or not task.startswith("task")
        or len(digest) != 64
        or not kaggle_ref.isdigit()
    ):
        raise argparse.ArgumentTypeError(
            "online exception must be taskXXX=<sha256>@<complete-kaggle-ref>"
        )
    return task, (digest.lower(), kaggle_ref)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one deterministic full-400 probe package.")
    parser.add_argument("--baseline-manifest", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--parent-dir", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--replace", type=parse_replacement, action="append", required=True)
    parser.add_argument(
        "--online-verified-exception",
        type=parse_online_exception,
        action="append",
        default=[],
        help="Permit one runtime audit exception only for an exact SHA already COMPLETE online.",
    )
    parser.add_argument("--message", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(HERE.parent))
    from c_score_common import score_onnx
    from candidate_registry import operator_audit
    from full400_safety import (
        TASKS,
        assert_complete_onnx_directory,
        atomic_write_json,
        deterministic_zip,
        load_baseline_manifest,
        model_hashes,
        model_set_sha256,
        sha256_file,
        verify_zip,
    )

    baseline = load_baseline_manifest(args.baseline_manifest, args.parent_dir)
    parent_dir = Path(baseline["onnx_dir"])
    replacements = dict(args.replace)
    online_exceptions = dict(args.online_verified_exception)
    unknown = sorted(set(replacements) - set(TASKS))
    if unknown:
        raise RuntimeError(f"unknown replacement tasks: {unknown}")
    output_root = args.output_root.resolve()
    output_onnx = output_root / "onnx"
    output_onnx.mkdir(parents=True, exist_ok=True)
    records = {}
    total_gain = 0.0
    for task in TASKS:
        parent = parent_dir / f"{task}.onnx"
        source = replacements.get(task, parent)
        if task in replacements:
            audit = operator_audit(source)
            parent_audit = operator_audit(parent)
            candidate_sha = sha256_file(source)
            exception = online_exceptions.get(task)
            exception_applied = bool(
                exception
                and exception[0] == candidate_sha
                and exception[1]
            )
            inherited_parent_risk = bool(
                not audit["runtime_compatible"]
                and audit.get("forbidden_ops") == parent_audit.get("forbidden_ops")
                and audit.get("negative_padding") == parent_audit.get("negative_padding")
            )
            if (
                not audit["runtime_compatible"]
                and not inherited_parent_risk
                and not exception_applied
            ):
                raise RuntimeError(f"unsafe replacement {task}: {audit}")
            parent_score = score_onnx(task, parent, validate_all=True)
            candidate_score = score_onnx(task, source, validate_all=True)
            if not (
                parent_score.ok
                and candidate_score.ok
                and candidate_score.examples_checked == candidate_score.examples_passed
                and candidate_score.cost is not None
                and parent_score.cost is not None
                and candidate_score.cost < parent_score.cost
            ):
                raise RuntimeError(
                    f"replacement is not fully valid and lower-cost for {task}: "
                    f"parent={asdict(parent_score)} candidate={asdict(candidate_score)}"
                )
            gain = math.log(parent_score.cost / candidate_score.cost)
            total_gain += gain
            records[task] = {
                "parent": asdict(parent_score),
                "candidate": asdict(candidate_score),
                "delta_points": gain,
                "operator_audit": audit,
                "parent_operator_audit": parent_audit,
                "inherited_parent_risk": inherited_parent_risk,
                "online_verified_exception": {
                    "applied": exception_applied,
                    "candidate_sha256": candidate_sha,
                    "kaggle_ref": exception[1] if exception_applied else None,
                },
            }
        shutil.copyfile(source, output_onnx / f"{task}.onnx")

    assert_complete_onnx_directory(output_onnx)
    hashes = model_hashes(output_onnx)
    first_zip = output_root / "submission.zip"
    second_zip = output_root / "submission.repeat.zip"
    first_sha = deterministic_zip(output_onnx, first_zip)
    second_sha = deterministic_zip(output_onnx, second_zip)
    if first_sha != second_sha:
        raise RuntimeError(f"non-deterministic package: {first_sha} != {second_sha}")
    second_zip.unlink()
    verification = verify_zip(first_zip, hashes)
    manifest = {
        "baseline": baseline,
        "message": args.message,
        "replacements": records,
        "replacement_count": len(records),
        "predicted_delta_points": total_gain,
        "predicted_public_score": float(baseline["public_score"]) + total_gain,
        "model_set_sha256": model_set_sha256(hashes),
        "package_sha256": first_sha,
        "deterministic_zip": True,
        "zip_verification": verification,
    }
    atomic_write_json(output_root / "package_manifest.json", manifest)
    print(json.dumps({
        "output_root": str(output_root),
        "package_sha256": first_sha,
        "replacement_count": len(records),
        "predicted_delta_points": total_gain,
        "predicted_public_score": manifest["predicted_public_score"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
