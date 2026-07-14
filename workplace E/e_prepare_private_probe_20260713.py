from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


EXPECTED_NAMES = [f"task{task:03d}.onnx" for task in range(1, 401)]
ZIP_TIMESTAMP = (2026, 7, 13, 0, 0, 0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="ascii")


def require_new_dir(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing directory: {path}")
    path.mkdir(parents=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a one-task cumulative private Kaggle probe."
    )
    parser.add_argument("--parent-models", type=Path, required=True)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--notebook-dir", type=Path, required=True)
    parser.add_argument("--dataset-slug", required=True)
    parser.add_argument("--dataset-title", required=True)
    parser.add_argument("--notebook-slug", required=True)
    parser.add_argument("--notebook-title", required=True)
    parser.add_argument("--parent-ref", type=int, required=True)
    parser.add_argument("--parent-score", type=float, required=True)
    parser.add_argument("--parent-dataset", required=True)
    parser.add_argument("--baseline-cost", type=int, required=True)
    parser.add_argument("--candidate-cost", type=int, required=True)
    parser.add_argument("--expected-gain", type=float, required=True)
    parser.add_argument("--method", required=True)
    args = parser.parse_args()

    if not 1 <= args.task <= 400:
        raise ValueError(f"task must be in 1..400, got {args.task}")

    parent_files = {path.name: path for path in args.parent_models.glob("task*.onnx")}
    if sorted(parent_files) != EXPECTED_NAMES:
        raise RuntimeError(
            f"parent must contain exactly task001-task400; found {len(parent_files)}"
        )

    target_name = f"task{args.task:03d}.onnx"
    if args.candidate_model.name != target_name:
        raise ValueError(
            f"candidate filename must be {target_name}, got {args.candidate_model.name}"
        )

    require_new_dir(args.package_dir)
    models_dir = args.package_dir / "models"
    models_dir.mkdir()
    for name in EXPECTED_NAMES:
        shutil.copy2(parent_files[name], models_dir / name)
    shutil.copy2(args.candidate_model, models_dir / target_name)

    changed = [
        name
        for name in EXPECTED_NAMES
        if sha256_file(parent_files[name]) != sha256_file(models_dir / name)
    ]
    if changed != [target_name]:
        raise RuntimeError(f"expected only {target_name} to change, got {changed}")

    submission = args.package_dir / "submission.zip"
    with zipfile.ZipFile(
        submission, "w", zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in EXPECTED_NAMES:
            info = zipfile.ZipInfo(name, ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (models_dir / name).read_bytes())

    with zipfile.ZipFile(submission) as archive:
        if sorted(archive.namelist()) != EXPECTED_NAMES:
            raise RuntimeError("submission ZIP does not contain 400 flat model entries")

    submission_sha256 = sha256_file(submission)

    require_new_dir(args.dataset_dir)
    shutil.copy2(submission, args.dataset_dir / "submission.zip")
    write_json(
        args.dataset_dir / "dataset-metadata.json",
        {
            "title": args.dataset_title,
            "id": f"llccqq624/{args.dataset_slug}",
            "licenses": [{"name": "MIT"}],
        },
    )
    write_json(
        args.dataset_dir / "provenance.json",
        {
            "competition": "neurogolf-2026",
            "target_parent_ref": args.parent_ref,
            "target_parent_public_score": args.parent_score,
            "parent_private_dataset": args.parent_dataset,
            "implementation": args.method,
            "changed_tasks": [args.task],
            "baseline_cost": args.baseline_cost,
            "candidate_cost": args.candidate_cost,
            "expected_local_gain": args.expected_gain,
            "submission_sha256": submission_sha256,
            "submission_entries": 400,
        },
    )

    require_new_dir(args.notebook_dir)
    write_json(
        args.notebook_dir / "kernel-metadata.json",
        {
            "id": f"llccqq624/{args.notebook_slug}",
            "title": args.notebook_title,
            "code_file": "submit.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": False,
            "enable_internet": False,
            "dataset_sources": [f"llccqq624/{args.dataset_slug}"],
            "competition_sources": ["neurogolf-2026"],
            "kernel_sources": [],
        },
    )
    (args.notebook_dir / "submit.py").write_text(
        """from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path


INPUT_ROOT = Path(\"/kaggle/input\")
OUTPUT = Path(\"/kaggle/working/submission.zip\")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open(\"rb\") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b\"\"):
            digest.update(chunk)
    return digest.hexdigest()


expected_names = [f\"task{task:03d}.onnx\" for task in range(1, 401)]
models = {path.name: path for path in INPUT_ROOT.rglob(\"task*.onnx\")}
if sorted(models) != expected_names:
    raise RuntimeError(
        f\"private input does not contain exactly task001-task400; found {len(models)}\"
    )

timestamp = (2026, 7, 13, 0, 0, 0)
with zipfile.ZipFile(OUTPUT, \"w\", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for name in expected_names:
        info = zipfile.ZipInfo(name, timestamp)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, models[name].read_bytes())

print(f\"wrote {OUTPUT} sha256={sha256_file(OUTPUT)} entries={len(models)}\")
""",
        encoding="ascii",
    )

    print(
        json.dumps(
            {
                "package": str(submission),
                "sha256": submission_sha256,
                "changed": changed,
                "dataset_dir": str(args.dataset_dir),
                "notebook_dir": str(args.notebook_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
