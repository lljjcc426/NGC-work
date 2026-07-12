from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def build(source: Path, output: Path) -> Path:
    model = deepcopy(onnx.load(source))
    old = list(model.graph.node)
    if old[0].op_type != "Einsum" or old[10].op_type != "Slice":
        raise RuntimeError("unexpected task091 graph")

    row_locator = [
        helper.make_node("Einsum", ["input", "egray"], ["grayrow"], equation="bchw,c->h", name="gray_row_projection"),
        helper.make_node("ArgMax", ["grayrow"], ["top_gray64"], axis=0, keepdims=0, select_last_index=0, name="first_gray_row"),
        helper.make_node("ArgMax", ["grayrow"], ["bot_gray64"], axis=0, keepdims=0, select_last_index=1, name="last_gray_row"),
        helper.make_node("Cast", ["top_gray64"], ["top_gray"], to=TensorProto.INT32, name="top_gray_to_i32"),
        helper.make_node("Cast", ["bot_gray64"], ["bot_gray"], to=TensorProto.INT32, name="bottom_gray_to_i32"),
    ]
    # Keep the left-column crop because it is reused as the output row mask,
    # but replace its endpoint decoder (11:17) with a direct row projection.
    del model.graph.node[:]
    model.graph.node.extend(old[:11] + row_locator + old[17:])

    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.source, args.output))


if __name__ == "__main__":
    main()
