from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import onnx


def build(source: Path, output: Path) -> Path:
    model = deepcopy(onnx.load(source))
    old = list(model.graph.node)
    if len(old) != 15 or old[8].op_type != "MaxPool" or old[9].op_type != "Min":
        raise RuntimeError("unexpected task077 graph")
    old[7].output[0] = "F3"
    del model.graph.node[:]
    model.graph.node.extend(old[:8] + old[10:])
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
