from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import helper


def prune_dead(model: onnx.ModelProto) -> int:
    live = {value.name for value in model.graph.output}
    kept: list[onnx.NodeProto] = []
    removed = 0
    for node in reversed(model.graph.node):
        if any(output and output in live for output in node.output):
            live.update(name for name in node.input if name)
            kept.append(node)
        else:
            removed += 1
    kept.reverse()
    del model.graph.node[:]
    model.graph.node.extend(kept)
    return removed


def prune_initializers(model: onnx.ModelProto) -> int:
    used = {name for node in model.graph.node for name in node.input if name}
    kept = [item for item in model.graph.initializer if item.name in used]
    removed = len(model.graph.initializer) - len(kept)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return removed


def rewrite(source: Path, destination: Path) -> None:
    model = onnx.load(source)
    nodes: list[onnx.NodeProto] = []
    replaced_distance = 0

    for original in model.graph.node:
        node = onnx.NodeProto.FromString(original.SerializeToString())
        for index, name in enumerate(node.input):
            if name == "top_cpr":
                node.input[index] = "top_mrow"
                replaced_distance += 1
            elif name == "bot_cpr":
                node.input[index] = "bot_mrow"
                replaced_distance += 1

        nodes.append(node)

    if replaced_distance != 2:
        raise RuntimeError(f"unexpected graph: distances={replaced_distance}")

    del model.graph.node[:]
    model.graph.node.extend(nodes)
    dead = prune_dead(model)
    unused = prune_initializers(model)
    live_outputs = {
        output for node in model.graph.node for output in node.output if output
    }
    kept_value_info = [
        onnx.ValueInfoProto.FromString(item.SerializeToString())
        for item in model.graph.value_info
        if item.name in live_outputs
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_value_info)
    onnx.checker.check_model(model, full_check=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)
    print(
        f"saved={destination} nodes={len(model.graph.node)} "
        f"dead_nodes={dead} unused_initializers={unused}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    rewrite(args.source, args.destination)


if __name__ == "__main__":
    main()
