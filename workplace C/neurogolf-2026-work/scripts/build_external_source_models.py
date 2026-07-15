from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build task ONNX files from a source-owned NeuroGolf repository."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--tasks", default="1-400")
    return parser.parse_args()


def parse_tasks(spec: str) -> list[int]:
    tasks: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(value) for value in part.split("-", 1))
            tasks.update(range(start, end + 1))
        else:
            tasks.add(int(part))
    return sorted(tasks)


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    os.environ["NEUROGOLF_ROOT"] = str(source_root)
    sys.path.insert(0, str(source_root / "src"))
    sys.path.insert(0, str(source_root))

    from src.harness import load_task

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for task_num in parse_tasks(args.tasks):
        task_name = f"task{task_num:03d}"
        output_path = output_dir / f"{task_name}.onnx"
        try:
            module = importlib.import_module(f"src.custom.{task_name}")
            model = module.build(load_task(task_num))
            if model is None:
                raise ValueError("builder returned None")
            onnx.checker.check_model(model, full_check=True)
            onnx.save_model(model, output_path)
            payload = output_path.read_bytes()
            rows.append(
                {
                    "task": task_name,
                    "ok": True,
                    "path": str(output_path),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "nodes": len(model.graph.node),
                    "initializers": len(model.graph.initializer),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "task": task_name,
                    "ok": False,
                    "error": f"{type(exc).__name__}:{exc}",
                }
            )
        if task_num % 25 == 0:
            print(json.dumps({"completed": task_num, "built": sum(r["ok"] for r in rows)}), flush=True)

    payload = {
        "source_root": str(source_root),
        "source_commit": _git_commit(source_root),
        "output_dir": str(output_dir),
        "built": sum(row["ok"] for row in rows),
        "failed": sum(not row["ok"] for row in rows),
        "rows": rows,
    }
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("built", "failed")}))
    return 0 if payload["built"] else 1


def _git_commit(root: Path) -> str | None:
    head = root / ".git" / "HEAD"
    if not head.exists():
        return None
    value = head.read_text(encoding="ascii").strip()
    if value.startswith("ref: "):
        ref = root / ".git" / value[5:]
        return ref.read_text(encoding="ascii").strip() if ref.exists() else None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
