from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("task349_width29", HERE / "build_width29_halo.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load build_width29_halo.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def build() -> Path:
    return MODULE.build(HERE.parent / "onnx" / "task349_candidate.onnx", trim_top=True)


if __name__ == "__main__":
    print(build())
