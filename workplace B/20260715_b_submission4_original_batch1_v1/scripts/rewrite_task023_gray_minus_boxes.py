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
    replaced_red = False

    for node in model.graph.node:
        outputs = set(node.output)

        if "paint_413" in outputs:
            # Every gray pixel is either a cyan 2x2 box or a red 1x3 stick.
            nodes.append(
                helper.make_node(
                    "BitwiseXor",
                    ["packed_rows_7", "paint_404"],
                    ["paint_413"],
                    name="task023_red_is_gray_minus_boxes",
                )
            )
            replaced_red = True
            continue

        nodes.append(node)

    if not replaced_red:
        raise RuntimeError("red paint anchor missing")

    del model.graph.node[:]
    model.graph.node.extend(nodes)

    dead = prune_dead(model)
    unused = prune_initializers(model)
    onnx.checker.check_model(model)
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
