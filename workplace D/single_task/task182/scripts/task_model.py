from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


DEFAULT_PARENT = Path(
    r"E:/kagglegolf/submissions/candidates/"
    r"GOLF_20260713_SUBMISSION8_REBASE/onnx/task182.onnx"
)
DEFAULT_OUTPUT = Path(__file__).parents[1] / "onnx" / "task182_candidate.onnx"


def build(source: Path, output: Path) -> Path:
    model = deepcopy(onnx.load(source))
    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    weights = initializers["cw"][:, :, :1, :1].copy()

    kept = [
        item
        for item in model.graph.initializer
        if item.name not in {"cw", "yzp_f16"}
    ]
    kept.extend(
        [
            numpy_helper.from_array(weights.astype(np.float32), "cw"),
            numpy_helper.from_array(
                np.ones((1, 1, 5, 5), dtype=np.uint8),
                "task182_count_kernel",
            ),
        ]
    )
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)

    rebuilt = []
    for node in model.graph.node:
        output_name = node.output[0]
        if output_name == "cidf32":
            rebuilt.append(
                helper.make_node(
                    "Conv",
                    ["input", "cw"],
                    ["cidf32"],
                    name="task182_cropped_color_projection",
                    pads=[0, 0, -10, -10],
                )
            )
        elif output_name in {"tmask_f16", "nshape_f16", "thr_f16"}:
            continue
        elif output_name == "thr_u8":
            rebuilt.extend(
                [
                    helper.make_node(
                        "QLinearConv",
                        [
                            "tmask_u8",
                            "xsc",
                            "z_u8",
                            "task182_count_kernel",
                            "xsc",
                            "z_u8",
                            "xsc",
                            "z_u8",
                        ],
                        ["task182_shape_count"],
                        name="task182_shape_count",
                    ),
                    helper.make_node(
                        "Add",
                        ["task182_shape_count", "yzp"],
                        ["thr_u8"],
                        name="task182_quantized_threshold",
                    ),
                ]
            )
        else:
            rebuilt.append(node)

    del model.graph.node[:]
    model.graph.node.extend(rebuilt)
    stale = {"tmask_f16", "nshape_f16", "thr_f16"}
    kept_value_info = [item for item in model.graph.value_info if item.name not in stale]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_value_info)
    model.graph.value_info.append(
        helper.make_tensor_value_info(
            "task182_shape_count", TensorProto.UINT8, [1, 1, 1, 1]
        )
    )
    model.producer_name = "ngc_task182_cropped_projection_quantized_count"
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(build(args.parent, args.output))


if __name__ == "__main__":
    main()
