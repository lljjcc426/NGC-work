from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("task349_width29", HERE / "build_width29_halo.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load build_width29_halo.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


if __name__ == "__main__":
    for side in ("top", "bottom"):
        output = HERE.parent / "onnx" / f"probe_width29_trim_{side}.onnx"
        kwargs = {f"trim_{side}": True}
        print(MODULE.build(output, **kwargs))
