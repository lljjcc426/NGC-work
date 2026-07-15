from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def build(parent_path: Path, output_path: Path) -> Path:
    model = onnx.load(parent_path)
    terminal = next(node for node in model.graph.node if node.op_type == "Einsum")
    attr = next(item for item in terminal.attribute if item.name == "equation")
    equation = attr.s.decode("ascii")
    lhs, rhs = equation.split("->", 1)
    terms = lhs.split(",")
    inputs = list(terminal.input)
    if inputs[:2] != ["G", "H"] or terms[:2] != ["abe", "abf"]:
        raise RuntimeError("unexpected task187 boundary selector inputs")

    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    ones = np.ones(25, dtype=np.float32)
    endpoints = np.zeros(25, dtype=np.float32)
    endpoints[[0, -1]] = 1.0
    basis = np.stack([ones, endpoints])

    g_coeff = np.zeros((2, 2, 2), dtype=np.float32)
    g_coeff[0, 0, 0] = 1.0
    g_coeff[1, 0, 1] = 1.0
    g_coeff[1, 1, 0] = 1.0
    h_coeff = np.zeros((2, 2, 2), dtype=np.float32)
    h_coeff[0, 0, 0] = 1.0
    h_coeff[1, 0, 0] = 1.0
    h_coeff[1, 1, 1] = 1.0

    if not np.array_equal(arrays["G"], np.einsum("abx,xe->abe", g_coeff, basis)):
        raise RuntimeError("G factorization is not exact")
    if not np.array_equal(arrays["H"], np.einsum("abw,wf->abf", h_coeff, basis)):
        raise RuntimeError("H factorization is not exact")

    terminal.input[:2] = ["G_coeff", "boundary_basis"]
    terminal.input.insert(2, "H_coeff")
    terminal.input.insert(3, "boundary_basis")
    attr.s = (",".join(["abx", "xe", "abw", "wf", *terms[2:]]) + "->" + rhs).encode("ascii")

    kept = [item for item in model.graph.initializer if item.name not in {"G", "H"}]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(g_coeff, name="G_coeff"),
            numpy_helper.from_array(h_coeff, name="H_coeff"),
            numpy_helper.from_array(basis, name="boundary_basis"),
        ]
    )
    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(inferred, output_path)
    reloaded = onnx.load(output_path)
    onnx.checker.check_model(reloaded, full_check=True)
    onnx.shape_inference.infer_shapes(reloaded, strict_mode=True, data_prop=True)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Factor task187 G/H selectors over a shared exact basis.")
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.parent, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
