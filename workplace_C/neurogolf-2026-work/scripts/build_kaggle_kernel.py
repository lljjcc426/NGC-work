from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = PROJECT_ROOT / "kaggle_kernel"


def build_kernel(output_file: str) -> None:
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    script_path = KERNEL_DIR / "neurogolf_2026_baseline.py"
    source = f'''from pathlib import Path
import sys

WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
PROJECT = WORKING / "neurogolf_2026_project"
PROJECT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT / "src"))

# Copy project source into the Kaggle working directory when this script is packaged
# with the src directory. The local build script places src next to this file.
LOCAL_SRC = Path(__file__).resolve().parent / "src"
if LOCAL_SRC.exists():
    import shutil
    target = PROJECT / "src"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(LOCAL_SRC, target)

raw = PROJECT / "data" / "raw"
raw.mkdir(parents=True, exist_ok=True)

# Link or copy Kaggle input files into the expected project layout.
for path in INPUT.rglob("*"):
    if path.is_file():
        rel = path.relative_to(INPUT)
        out = raw / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            try:
                out.symlink_to(path)
            except Exception:
                shutil.copy2(path, out)

sys.path.insert(0, str(PROJECT))
from src.make_baseline import main as make_baseline_main
from src.validate_submission import main as validate_main

make_baseline_main()
submission = PROJECT / "submissions" / "submission_baseline.csv"
final_path = WORKING / "{output_file}"
if submission.exists():
    shutil.copy2(submission, final_path)
else:
    raise SystemExit("Baseline did not create a submission file.")
print(f"Wrote {{final_path}}")
'''
    script_path.write_text(source, encoding="utf-8")

    bundled_src = KERNEL_DIR / "src"
    if bundled_src.exists():
        shutil.rmtree(bundled_src)
    shutil.copytree(PROJECT_ROOT / "src", bundled_src, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    metadata = {
        "id": "whzy3185/neurogolf-2026-baseline",
        "title": "neurogolf-2026 baseline",
        "code_file": script_path.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": ["neurogolf-2026"],
        "kernel_sources": [],
      }
    (KERNEL_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {script_path}")
    print(f"Wrote {KERNEL_DIR / 'kernel-metadata.json'}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-file", default="submission.csv", help="Kaggle working output file name.")
    args = parser.parse_args()
    build_kernel(args.output_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
