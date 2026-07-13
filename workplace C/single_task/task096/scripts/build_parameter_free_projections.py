from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import onnx
from onnx import helper


TASK_ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(
    r"E:\kagglegolf\submissions\candidates\GOLF_20260713_SUBMISSION8_REBASE\onnx\task096.onnx"
)


def build(output: Path) -> Path:
    model = deepcopy(onnx.load(SOURCE))
    nodes = list(model.graph.node)
    if nodes[0].op_type != "Conv" or nodes[47].op_type != "Conv":
        raise RuntimeError("unexpected task096 projection graph")

    nodes[0].CopyFrom(
        helper.make_node(
            "LpPool",
            ["input"],
            ["row_sum_compact"],
            kernel_shape=[1, 19],
            pads=[0, 0, -11, 0],
            strides=[1, 30],
            p=1,
        )
    )
    nodes[1].input[:] = ["row_sum_compact", "compact_count_axes"]
    nodes[7].input[:] = ["row_sum_compact"]
    nodes[47].CopyFrom(
        helper.make_node(
            "LpPool",
            ["input"],
            ["col_sum_compact"],
            kernel_shape=[19, 1],
            pads=[0, 0, 0, -11],
            strides=[30, 1],
            p=1,
        )
    )
    nodes[48].input[:] = ["col_sum_compact"]
    del model.graph.node[:]
    model.graph.node.extend(nodes)

    used = {name for node in nodes for name in node.input}
    kept = [value for value in model.graph.initializer if value.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.name = "task096_parameter_free_row_col_projection"
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)
    return output


if __name__ == "__main__":
    print(build(TASK_ROOT / "onnx" / "task096_parameter_free_projections.onnx"))
