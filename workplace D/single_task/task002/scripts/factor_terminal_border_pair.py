from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def _einsum_equation(node: onnx.NodeProto) -> onnx.AttributeProto:
    return next(attr for attr in node.attribute if attr.name == "equation")


def build(source: Path, output: Path) -> Path:
    model = onnx.load(source)
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    if "G" not in arrays or "H" not in arrays:
        raise RuntimeError("expected G and H initializers")

    g = arrays["G"]
    h = arrays["H"]
    if g.shape != (4, 20) or h.shape != (4, 20):
        raise RuntimeError(f"unexpected G/H shapes: {g.shape}, {h.shape}")

    border = np.zeros(20, dtype=g.dtype)
    border[[0, -1]] = 1
    left_basis = np.stack([np.ones(20, dtype=g.dtype), border])
    right_basis = np.stack([border, np.ones(20, dtype=g.dtype)])
    original_pair = g.T.astype(np.float32) @ h.astype(np.float32)
    factored_pair = left_basis.T.astype(np.float32) @ right_basis.astype(np.float32)
    if not np.array_equal(original_pair, factored_pair):
        raise RuntimeError("border factorization is not exact")

    matches = [node for node in model.graph.node if node.op_type == "Einsum" and list(node.input[:2]) == ["G", "H"]]
    if len(matches) != 1:
        raise RuntimeError(f"expected one G/H Einsum consumer, found {len(matches)}")
    node = matches[0]
    equation = _einsum_equation(node)
    lhs, rhs = equation.s.decode("ascii").split("->", 1)
    terms = lhs.split(",")
    if terms[:2] != ["ad", "ae"]:
        raise RuntimeError(f"unexpected leading Einsum terms: {terms[:2]}")

    # The source equation already uses all 52 ASCII Einsum labels, so retain
    # its contracted ``a`` axis and reduce that dimension from four to two.
    node.input[:2] = ["BORDER_LEFT", "BORDER_RIGHT"]
    equation.s = (",".join(terms) + "->" + rhs).encode("ascii")

    kept = [item for item in model.graph.initializer if item.name not in {"G", "H"}]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(left_basis, "BORDER_LEFT"),
            numpy_helper.from_array(right_basis, "BORDER_RIGHT"),
        ]
    )
    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(inferred, output)

    reloaded = onnx.load(output)
    onnx.checker.check_model(reloaded, full_check=True)
    onnx.shape_inference.infer_shapes(reloaded, strict_mode=True, data_prop=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Factor task002 terminal G/H border pair exactly.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    path = build(args.source, args.output)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
