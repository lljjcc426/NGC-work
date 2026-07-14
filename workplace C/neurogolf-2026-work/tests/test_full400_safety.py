from __future__ import annotations

import json
from pathlib import Path

from full400_safety import TASKS, deterministic_zip, verify_zip


def test_deterministic_zip(tmp_path: Path) -> None:
    source = tmp_path / "onnx"
    source.mkdir()
    for index, task in enumerate(TASKS):
        (source / f"{task}.onnx").write_bytes(f"model-{index}".encode())
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    assert deterministic_zip(source, first) == deterministic_zip(source, second)
    assert verify_zip(first)["model_count"] == 400


def test_zip_rejects_extra_onnx(tmp_path: Path) -> None:
    source = tmp_path / "onnx"
    source.mkdir()
    for task in TASKS:
        (source / f"{task}.onnx").write_bytes(b"x")
    (source / "task401.onnx").write_bytes(b"x")
    try:
        deterministic_zip(source, tmp_path / "bad.zip")
    except RuntimeError as exc:
        assert "invalid 400-model directory" in str(exc)
    else:
        raise AssertionError("extra ONNX must be rejected")
