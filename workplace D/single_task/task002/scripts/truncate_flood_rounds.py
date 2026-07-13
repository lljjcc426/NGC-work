from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import onnx


TRUNCATION_STATES = {
    1: "safe_name_157",
    2: "safe_name_148",
    3: "safe_name_139",
    4: "safe_name_130",
}


def prune_to_outputs(model: onnx.ModelProto) -> None:
    required = {item.name for item in model.graph.output}
    kept = []
    for node in reversed(model.graph.node):
        if any(name in required for name in node.output):
            kept.append(node)
            required.update(name for name in node.input if name)
    kept.reverse()
    del model.graph.node[:]
    model.graph.node.extend(kept)

    kept_initializers = [item for item in model.graph.initializer if item.name in required]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)


def build(source: Path, output: Path, removed_rounds: int) -> Path:
    model = deepcopy(onnx.load(source))
    replacement = TRUNCATION_STATES[removed_rounds]
    xor_node = next(
        node
        for node in model.graph.node
        if node.op_type == "BitwiseXor" and list(node.output) == ["safe_name_167"]
    )
    xor_node.input[1] = replacement
    prune_to_outputs(model)

    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--removed-rounds", type=int, choices=sorted(TRUNCATION_STATES), required=True)
    args = parser.parse_args()
    print(build(args.source, args.output, args.removed_rounds))


if __name__ == "__main__":
    main()
