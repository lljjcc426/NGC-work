from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


DEFAULT_SOURCE = Path(
    r"E:/kongming/NGC-work/workplace C/artifacts/"
    r"full400_round4_stack_task011_task324/onnx/task367.onnx"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "onnx" / "task367_reduce_max.onnx"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    model = onnx.load(args.source)
    nodes = list(model.graph.node)
    target = next(
        node
        for node in nodes
        if node.op_type == "QLinearConv" and list(node.output) == ["v_col"]
    )
    target_index = nodes.index(target)
    replacement = helper.make_node(
        "ReduceMax",
        ["v_main", "task367_channel_axis"],
        ["v_col"],
        name="task367_channel_or",
        keepdims=1,
    )
    nodes[target_index] = replacement
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.graph.initializer.append(
        numpy_helper.from_array(np.asarray([1], dtype=np.int64), name="task367_channel_axis")
    )
    consumers = {name for node in model.graph.node for name in node.input if name}
    kept = [tensor for tensor in model.graph.initializer if tensor.name in consumers]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.producer_name = "ngc_task367_exact_channel_or"
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(inferred, full_check=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(inferred, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
