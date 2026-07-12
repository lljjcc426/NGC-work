from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("task349_collision_merge", HERE / "build_double_collision_merge.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load build_double_collision_merge.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def build() -> Path:
    return MODULE.build(HERE.parent / "onnx" / "task349_candidate.onnx")


if __name__ == "__main__":
    print(build())
