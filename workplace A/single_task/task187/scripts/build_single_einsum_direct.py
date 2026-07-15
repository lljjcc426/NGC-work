from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def _initializer_map(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}


def _pad_internal_state(value: np.ndarray, size: int = 30) -> np.ndarray:
    shape = list(value.shape)
    if shape[-1] != 25:
        raise RuntimeError(f"unexpected internal width: {value.shape}")
    shape[-1] = size
    if value.ndim >= 2 and value.shape[-2] == 25:
        shape[-2] = size
    result = np.zeros(shape, dtype=value.dtype)
    slices = tuple(slice(0, dim) for dim in value.shape)
    result[slices] = value
    return result


def build(parent_path: Path, output_path: Path) -> Path:
    model = onnx.load(parent_path)
    if [node.op_type for node in model.graph.node] != ["Conv", "Einsum"]:
        raise RuntimeError("expected the safe task187 Conv + terminal Einsum graph")

    conv, terminal = model.graph.node
    if list(conv.output) != ["t2"] or list(conv.input) != ["input", "Wt", "Bt"]:
        raise RuntimeError("unexpected task187 crop Conv")
    equation_attr = next(attr for attr in terminal.attribute if attr.name == "equation")
    equation = equation_attr.s.decode("ascii")
    lhs, rhs = equation.split("->", 1)
    terms = lhs.split(",")
    inputs = list(terminal.input)
    if len(terms) != len(inputs):
        raise RuntimeError("Einsum input/equation arity mismatch")

    arrays = _initializer_map(model)
    expanded = {
        name: _pad_internal_state(arrays[name])
        for name in ("G", "H", "S")
    }
    expanded["T"] = arrays["T"]
    expanded["background_channel"] = np.array(
        [1.0] + [0.0] * 9,
        dtype=np.float32,
    )

    new_inputs: list[str] = ["background_channel"]
    new_terms: list[str] = ["x"]
    for name, term in zip(inputs, terms, strict=True):
        if name == "P":
            continue
        if name == "t2":
            if not term.startswith("wx"):
                raise RuntimeError(f"unexpected t2 term: {term}")
            name = "input"
            term = "...x" + term[2:]
        # P was an identity embedding from the 25x25 internal state into the
        # top-left of the 30x30 output. Expanded zero-padded state tensors let
        # the terminal indices become output indices directly.
        term = term.replace("W", "Y").replace("X", "Z")
        new_inputs.append(name)
        new_terms.append(term)

    new_equation = ",".join(new_terms) + "->" + rhs
    direct = helper.make_node(
        "Einsum",
        new_inputs,
        list(terminal.output),
        name="task187_single_einsum_direct",
        equation=new_equation,
    )

    del model.graph.node[:]
    model.graph.node.append(direct)
    del model.graph.initializer[:]
    for name in ("G", "H", "S", "T", "background_channel"):
        model.graph.initializer.append(numpy_helper.from_array(expanded[name], name=name))
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
    parser = argparse.ArgumentParser(
        description="Fold task187's crop Conv and terminal projection into one direct Einsum."
    )
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.parent, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
