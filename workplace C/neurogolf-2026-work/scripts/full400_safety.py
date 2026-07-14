from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any


TASKS = tuple(f"task{index:03d}" for index in range(1, 401))
FIXED_ZIP_TIMESTAMP = (2020, 1, 1, 0, 0, 0)
ALLOWED_CANDIDATE_STATES = {"runtime_safe", "canonical", "online_verified"}
FORBIDDEN_OPERATORS = {"TopK:uint8"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(raw_temp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_complete_onnx_directory(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(path)
    root_files = sorted(item.name for item in path.iterdir() if item.is_file())
    onnx_files = [name for name in root_files if name.lower().endswith(".onnx")]
    expected = [f"{task}.onnx" for task in TASKS]
    if onnx_files != expected:
        missing = sorted(set(expected) - set(onnx_files))
        extra = sorted(set(onnx_files) - set(expected))
        raise RuntimeError(
            f"invalid 400-model directory: count={len(onnx_files)} "
            f"missing={missing[:10]} extra={extra[:10]}"
        )


def model_hashes(path: Path) -> dict[str, str]:
    assert_complete_onnx_directory(path)
    return {task: sha256_file(path / f"{task}.onnx") for task in TASKS}


def model_set_sha256(hashes: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for task in TASKS:
        digest.update(task.encode("ascii"))
        digest.update(b"\0")
        digest.update(hashes[task].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def deterministic_zip(source_dir: Path, destination: Path, compresslevel: int = 9) -> str:
    assert_complete_onnx_directory(source_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    with zipfile.ZipFile(
        destination,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=compresslevel,
        strict_timestamps=True,
    ) as archive:
        for task in TASKS:
            source = source_dir / f"{task}.onnx"
            info = zipfile.ZipInfo(source.name, FIXED_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.flag_bits = 0
            archive.writestr(info, source.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=compresslevel)
    return sha256_file(destination)


def verify_zip(zip_path: Path, expected_hashes: dict[str, str] | None = None) -> dict[str, Any]:
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        expected_names = [f"{task}.onnx" for task in TASKS]
        if names != expected_names:
            raise RuntimeError(f"zip root entries are not deterministic task001-task400: count={len(names)}")
        hashes = {
            Path(name).stem: sha256_bytes(archive.read(name))
            for name in names
        }
    if expected_hashes is not None and hashes != expected_hashes:
        mismatches = [task for task in TASKS if hashes.get(task) != expected_hashes.get(task)]
        raise RuntimeError(f"zip model SHA mismatch: {mismatches[:10]}")
    return {
        "package_sha256": sha256_file(zip_path),
        "model_set_sha256": model_set_sha256(hashes),
        "model_count": len(hashes),
        "models": hashes,
    }


def load_baseline_manifest(path: Path, parent_dir: Path | None = None) -> dict[str, Any]:
    manifest = read_json(path)
    required = {
        "name",
        "kaggle_ref",
        "public_score",
        "local_score",
        "onnx_dir",
        "package_sha256",
        "model_set_sha256",
        "model_count",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise RuntimeError(f"baseline manifest missing fields: {missing}")
    directory = (parent_dir or Path(manifest["onnx_dir"])).resolve()
    assert_complete_onnx_directory(directory)
    hashes = model_hashes(directory)
    actual_set_sha = model_set_sha256(hashes)
    if actual_set_sha != manifest["model_set_sha256"]:
        raise RuntimeError(
            f"parent model set SHA mismatch: manifest={manifest['model_set_sha256']} actual={actual_set_sha}"
        )
    if int(manifest["model_count"]) != 400:
        raise RuntimeError(f"baseline model_count must be 400, got {manifest['model_count']}")
    result = dict(manifest)
    result["onnx_dir"] = str(directory)
    result["models"] = hashes
    return result


def environment_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for module_name in ("onnx", "onnxruntime"):
        try:
            module = __import__(module_name)
            versions[module_name] = str(module.__version__)
        except Exception:
            versions[module_name] = "unavailable"
    return versions


def validation_cache_key(
    *,
    task: str,
    model_path: Path,
    task_json: Path,
    scorer_source: Path,
    validation_mode: str,
    max_examples: int,
) -> str:
    versions = environment_versions()
    payload = {
        "task": task,
        "model_sha256": sha256_file(model_path),
        "task_json_sha256": sha256_file(task_json),
        "scorer_source_sha256": sha256_file(scorer_source),
        "onnx_version": versions["onnx"],
        "onnxruntime_version": versions["onnxruntime"],
        "validation_mode": validation_mode,
        "max_examples": int(max_examples),
    }
    return sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
