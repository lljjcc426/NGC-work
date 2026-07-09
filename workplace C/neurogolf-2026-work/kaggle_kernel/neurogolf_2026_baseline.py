from pathlib import Path
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
final_path = WORKING / "submission.csv"
if submission.exists():
    shutil.copy2(submission, final_path)
else:
    raise SystemExit("Baseline did not create a submission file.")
print(f"Wrote {final_path}")
