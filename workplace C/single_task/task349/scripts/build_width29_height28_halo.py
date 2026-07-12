from pathlib import Path
import importlib.util


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("task349_width29", HERE / "build_width29_halo.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load build_width29_halo.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


if __name__ == "__main__":
    output = HERE.parent / "onnx" / "task349_width29_height28_halo.onnx"
    print(MODULE.build(output, trim_boundary_rows=True))
