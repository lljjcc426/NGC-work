from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from c_score_common import SCORE_DOCS, ensure_dirs, write_md


def list_names(path: Path) -> list[str]:
    if path.is_dir():
        return sorted(p.name for p in path.glob("task*.onnx"))
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            return sorted(Path(n).name for n in z.namelist() if Path(n).name.startswith("task") and Path(n).suffix == ".onnx")
    raise ValueError(f"candidate must be a directory or zip: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", default="")
    parser.add_argument("--candidate-zip", default="")
    args = parser.parse_args()
    ensure_dirs()
    path = Path(args.candidate_dir or args.candidate_zip)
    names = list_names(path)
    expected = [f"task{i:03d}.onnx" for i in range(1, 401)]
    missing = [x for x in expected if x not in names]
    extra = [x for x in names if x not in expected]
    duplicate_count = len(names) - len(set(names))
    ok = len(names) == 400 and not missing and not extra and duplicate_count == 0
    lines = [
        "# Candidate Validation Report",
        "",
        f"- candidate: `{path}`",
        f"- file_count: `{len(names)}`",
        f"- missing_task_count: `{len(missing)}`",
        f"- extra_task_count: `{len(extra)}`",
        f"- duplicate_count: `{duplicate_count}`",
        f"- validation_status: `{'VALID_STRUCTURE_400' if ok else 'INVALID_STRUCTURE'}`",
        "",
        f"- missing_head: `{', '.join(missing[:20])}`",
        f"- extra_head: `{', '.join(extra[:20])}`",
    ]
    out = SCORE_DOCS / "27_BATCH_VALIDATION_PLAN.md"
    write_md(out, "\n".join(lines))
    print(out)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
