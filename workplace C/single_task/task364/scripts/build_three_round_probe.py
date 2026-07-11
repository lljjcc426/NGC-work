from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import onnx


def build(source: Path, output: Path) -> Path:
    model = deepcopy(onnx.load(source))
    nodes = list(model.graph.node)
    if len(nodes) != 27 or nodes[24].op_type != "Add" or nodes[24].input[0] != "SE5":
        raise RuntimeError("unexpected task364 source graph")

    # Keep three masked 3x3 propagation rounds and bypass rounds four/five.
    nodes[24].input[0] = "SE3"
    del nodes[20:24]
    del model.graph.node[:]
    model.graph.node.extend(nodes)

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
