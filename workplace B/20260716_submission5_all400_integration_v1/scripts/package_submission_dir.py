from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    expected = [f"task{task:03d}.onnx" for task in range(1, 401)]
    actual = sorted(path.name for path in args.submission_dir.glob("task*.onnx"))
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise SystemExit(f"invalid model set: missing={missing} extra={extra}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in expected:
            path = args.submission_dir / name
            info = zipfile.ZipInfo(name, date_time=(2026, 7, 16, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)

    with zipfile.ZipFile(args.output) as archive:
        names = archive.namelist()
        crc_errors = archive.testzip()
    if names != expected or crc_errors is not None:
        raise SystemExit(f"ZIP verification failed: crc_error={crc_errors}")

    result = {
        "submission_dir": str(args.submission_dir.resolve()),
        "output": str(args.output.resolve()),
        "models": len(names),
        "bytes": args.output.stat().st_size,
        "sha256": sha256(args.output),
        "crc_clean": True,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
