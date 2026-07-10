#!/usr/bin/env python
"""Build a submission package from a verified base and explicit task overrides."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import zipfile

import onnx


EXPECTED_NAMES = [f"task{task:03d}.onnx" for task in range(1, 401)]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_override(value: str) -> tuple[str, pathlib.Path]:
    try:
        task_name, path_text = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("override must be taskNNN=PATH") from exc
    if task_name not in EXPECTED_NAMES:
        raise argparse.ArgumentTypeError(f"invalid task name: {task_name}")
    path = pathlib.Path(path_text)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"override file does not exist: {path}")
    return task_name, path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument(
        "--override", type=parse_override, action="append", default=[], required=True
    )
    args = parser.parse_args()

    if not args.base.is_file():
        raise FileNotFoundError(args.base)
    overrides: dict[str, tuple[pathlib.Path, bytes]] = {}
    for task_name, path in args.override:
        if task_name in overrides:
            raise ValueError(f"duplicate override: {task_name}")
        payload = path.read_bytes()
        model = onnx.load_from_string(payload)
        onnx.checker.check_model(model)
        overrides[task_name] = (path, payload)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.base) as source:
        if sorted(source.namelist()) != EXPECTED_NAMES:
            raise RuntimeError("base zip does not contain exactly task001-task400")
        with zipfile.ZipFile(args.output, "w") as target:
            for info in source.infolist():
                payload = overrides.get(info.filename, (None, source.read(info.filename)))[1]
                target.writestr(info, payload)

    with zipfile.ZipFile(args.output) as check:
        if check.testzip() is not None:
            raise RuntimeError("output zip failed CRC validation")
        if sorted(check.namelist()) != EXPECTED_NAMES:
            raise RuntimeError("output zip task inventory mismatch")
        for task_name, (_, payload) in overrides.items():
            if sha256_bytes(check.read(task_name)) != sha256_bytes(payload):
                raise RuntimeError(f"embedded override mismatch: {task_name}")

    manifest = {
        "base": str(args.base.resolve()),
        "base_sha256": sha256_file(args.base),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "output_bytes": args.output.stat().st_size,
        "zip_entries": 400,
        "zip_test": None,
        "overrides": [
            {
                "task": task_name,
                "path": str(path.resolve()),
                "sha256": sha256_bytes(payload),
                "bytes": len(payload),
            }
            for task_name, (path, payload) in sorted(overrides.items())
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
