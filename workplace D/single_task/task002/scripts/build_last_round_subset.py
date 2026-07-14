from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import onnx
from onnx import helper


SOURCE_OUTPUTS = {
    "L": "safe_name_158",
    "R": "safe_name_159",
    "U": "safe_name_160",
    "D": "safe_name_161",
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
    initializers = [item for item in model.graph.initializer if item.name in required]
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)


def build(source: Path, output: Path, directions: str) -> Path:
    selected = [name for name in "LRUD" if name in directions]
    if not selected:
        raise ValueError("at least one direction is required")
    model = deepcopy(onnx.load(source))
    original_nodes = list(model.graph.node)
    by_output = {node.output[0]: node for node in original_nodes if node.output}

    # Retain only the selected shift/gather producers from the final round.
    additions = [deepcopy(by_output[SOURCE_OUTPUTS[name]]) for name in selected]
    values = ["safe_name_157", *(SOURCE_OUTPUTS[name] for name in selected)]
    current = values[0]
    for index, value in enumerate(values[1:]):
        merged = f"task002_last_{directions}_or_{index}"
        additions.append(helper.make_node("BitwiseOr", [current, value], [merged]))
        current = merged
    additions.append(
        helper.make_node(
            "BitwiseAnd", [current, "safe_name_25"], [f"task002_last_{directions}"]
        )
    )

    xor_node = deepcopy(by_output["safe_name_167"])
    xor_node.input[1] = f"task002_last_{directions}"
    prefix = [node for node in original_nodes if node.output[0] not in {
        "safe_name_158", "safe_name_159", "safe_name_160", "safe_name_161",
        "safe_name_162", "safe_name_163", "safe_name_164", "safe_name_165",
        "safe_name_166", "safe_name_167",
    }]
    # Insert the replacement immediately before the output-side suffix.
    suffix_start = next(i for i, node in enumerate(prefix) if node.output[0] == "safe_name_168")
    nodes = prefix[:suffix_start] + additions + [xor_node] + prefix[suffix_start:]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    prune_to_outputs(model)

    model.graph.name = f"task002_last_round_{directions}"
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, output)
    onnx.checker.check_model(onnx.load(output), full_check=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--directions", required=True)
    args = parser.parse_args()
    print(build(args.source, args.output, args.directions.upper()))


if __name__ == "__main__":
    main()
