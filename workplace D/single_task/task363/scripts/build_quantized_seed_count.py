from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


TASK_ROOT = Path(__file__).resolve().parents[1]
SOURCE = TASK_ROOT / "onnx" / "task363_candidate.onnx"


def build(output: Path) -> Path:
    model = onnx.load(SOURCE)
    nodes = list(model.graph.node)
    if [nodes[index].output[0] for index in (17, 18, 19)] != [
        "ker16",
        "seed_count",
        "seed_count_u8",
    ]:
        raise RuntimeError("unexpected task363 seed-count path")

    seed_count = helper.make_node(
        "QLinearConv",
        [
            "ker_u8",
            "task363_qscale",
            "task363_qzero",
            "task363_seed_weight",
            "task363_qscale",
            "task363_qzero",
            "task363_qscale",
            "task363_qzero",
        ],
        ["seed_count_u8"],
        kernel_shape=[4, 4],
    )
    nodes[17:20] = [seed_count]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.graph.initializer.append(
        numpy_helper.from_array(
            np.ones((1, 1, 4, 4), dtype=np.uint8),
            name="task363_seed_weight",
        )
    )
    kept = [
        value
        for value in model.graph.value_info
        if value.name not in {"ker16", "seed_count", "seed_count_u8"}
    ]
    kept.append(
        helper.make_tensor_value_info(
            "seed_count_u8", TensorProto.UINT8, [1, 1, 1, 1]
        )
    )
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept)
    model.graph.name = "task363_quantized_seed_count"
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)
    return output


if __name__ == "__main__":
    print(build(TASK_ROOT / "onnx" / "task363_quantized_seed_count.onnx"))
