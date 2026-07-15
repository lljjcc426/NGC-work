from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract selected ONNX tasks from ZIP files, deduplicated by SHA256."
    )
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-zip-mib", type=int, default=256)
    parser.add_argument("--score-examples", type=int, default=0)
    return parser.parse_args()


def zip_paths(roots: list[Path], max_bytes: int) -> list[Path]:
    found: set[Path] = set()
    for root in roots:
        if root.is_file() and root.suffix.lower() == ".zip":
            candidates = [root]
        elif root.is_dir():
            candidates = root.rglob("*.zip")
        else:
            continue
        for path in candidates:
            try:
                if path.stat().st_size <= max_bytes:
                    found.add(path.resolve())
            except OSError:
                continue
    return sorted(found, key=lambda path: str(path).lower())


def main() -> int:
    args = parse_args()
    tasks = {task if task.startswith("task") else f"task{int(task):03d}" for task in args.task}
    output = args.output.resolve()
    model_dir = output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    records: dict[tuple[str, str], dict] = {}
    errors: list[dict[str, str]] = []
    archives = zip_paths(args.root, args.max_zip_mib * 1024 * 1024)

    for index, archive in enumerate(archives, start=1):
        try:
            with zipfile.ZipFile(archive) as zf:
                for info in zf.infolist():
                    basename = Path(info.filename.replace("\\", "/")).name.lower()
                    if not basename.endswith(".onnx"):
                        continue
                    task = basename.removesuffix(".onnx")
                    if task not in tasks:
                        continue
                    payload = zf.read(info)
                    digest = hashlib.sha256(payload).hexdigest()
                    key = (task, digest)
                    if key not in records:
                        task_dir = model_dir / task
                        task_dir.mkdir(parents=True, exist_ok=True)
                        target = task_dir / f"{digest}.onnx"
                        target.write_bytes(payload)
                        records[key] = {
                            "task": task,
                            "sha256": digest,
                            "size": len(payload),
                            "model_path": str(target),
                            "sources": [],
                        }
                    records[key]["sources"].append(
                        {"archive": str(archive), "entry": info.filename}
                    )
        except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
            errors.append({"archive": str(archive), "error": f"{type(exc).__name__}:{exc}"})
        if index % 100 == 0:
            print(json.dumps({"archives": index, "unique_models": len(records)}), flush=True)

    rows = sorted(records.values(), key=lambda row: (row["task"], row["size"], row["sha256"]))
    if args.score_examples:
        sys.path.insert(0, str(HERE.parent))
        from c_score_common import score_onnx

        for index, row in enumerate(rows, start=1):
            result = score_onnx(
                row["task"],
                Path(row["model_path"]),
                validate_all=False,
                max_examples=args.score_examples,
            )
            row["score"] = result.__dict__
            if index % 25 == 0:
                print(json.dumps({"scored": index, "total": len(rows)}), flush=True)

    payload = {
        "roots": [str(path.resolve()) for path in args.root],
        "tasks": sorted(tasks),
        "archives_scanned": len(archives),
        "unique_models": len(rows),
        "models": rows,
        "errors": errors,
    }
    (output / "scan.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "archives_scanned": len(archives),
        "unique_models": len(rows),
        "errors": len(errors),
        "output": str(output / "scan.json"),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
