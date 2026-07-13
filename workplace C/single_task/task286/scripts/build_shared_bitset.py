from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


TASK_DIR = Path(__file__).resolve().parent.parent
ONNX_DIR = TASK_DIR / "onnx"
DEFAULT_PARENT = Path(
    r"E:\kagglegolf\submissions\candidates\GOLF_20260711_093_v92_plus_task349_k11\onnx\task286.onnx"
)


def build(source: Path, rounds: int, output_path: Path) -> None:
    model = onnx.load(source)
    graph = model.graph

    # Nodes 0..26 decode the grid and pack passable/seed cells into one uint32
    # per row. Keep that task-specific frontend and replace only its scalar
    # propagation network with a shared [1, 1, 25, 1] state.
    nodes = list(graph.node[:27])
    nodes.append(helper.make_node("Cast", ["pk2_s012"], ["shared_seed"], to=TensorProto.UINT32))

    up_indices = np.array([0, *range(24)], dtype=np.int64)
    down_indices = np.array([*range(1, 25), 24], dtype=np.int64)
    extra_initializers = [
        numpy_helper.from_array(up_indices, name="shared_up_indices"),
        numpy_helper.from_array(down_indices, name="shared_down_indices"),
    ]

    state = "shared_seed"
    for step in range(rounds):
        prefix = f"shared_r{step:03d}"
        left = f"{prefix}_left"
        right = f"{prefix}_right"
        horizontal = f"{prefix}_horizontal"
        horizontal_both = f"{prefix}_horizontal_both"
        up = f"{prefix}_up"
        down = f"{prefix}_down"
        vertical = f"{prefix}_vertical"
        expanded = f"{prefix}_expanded"
        next_state = f"{prefix}_state"
        nodes.extend(
            [
                helper.make_node("BitShift", [state, "S1"], [left], direction="LEFT"),
                helper.make_node("BitShift", [state, "S1"], [right], direction="RIGHT"),
                helper.make_node("BitwiseOr", [state, left], [horizontal]),
                helper.make_node("BitwiseOr", [horizontal, right], [horizontal_both]),
                helper.make_node("Gather", [state, "shared_up_indices"], [up], axis=2),
                helper.make_node("Gather", [state, "shared_down_indices"], [down], axis=2),
                helper.make_node("BitwiseOr", [up, down], [vertical]),
                helper.make_node("BitwiseOr", [horizontal_both, vertical], [expanded]),
                helper.make_node("BitwiseAnd", [expanded, "v14"], [next_state]),
            ]
        )
        state = next_state

    # Reuse the parent's unpacking and output-color logic. v4677 was the old
    # scalar network reshaped to exactly the same row-bitset shape as state.
    for old_node in graph.node[2382:]:
        node = onnx.NodeProto()
        node.CopyFrom(old_node)
        for index, name in enumerate(node.input):
            if name == "v4677":
                node.input[index] = state
        nodes.append(node)

    required = {name for node in nodes for name in node.input}
    initializers = [init for init in graph.initializer if init.name in required]
    initializers.extend(extra_initializers)
    del graph.node[:]
    graph.node.extend(nodes)
    del graph.initializer[:]
    graph.initializer.extend(initializers)

    graph.name = f"task286_shared_bitset_{rounds}_rounds"
    model.doc_string = (
        "task286-specific shared row-bitset flood fill; "
        f"{rounds} synchronous 4-neighbor propagation rounds"
    )
    onnx.checker.check_model(model, full_check=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ONNX_DIR / f"task286_shared_r{args.rounds}.onnx"
    build(args.source, args.rounds, output)
    print(output)


if __name__ == "__main__":
    main()
