from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


SCRIPT_DIR = Path(__file__).resolve().parent
TASK_DIR = SCRIPT_DIR.parent
DEFAULT_BASELINE = Path(
    r"E:\kagglegolf\submissions\candidates\GOLF_20260709_101_prvsiyan_7266_72_repro\onnx\task158.onnx"
)
DEFAULT_OUTPUT = TASK_DIR / "onnx" / "task158_candidate.onnx"


def replace_scaled_stamp_gathers(model: onnx.ModelProto) -> onnx.ModelProto:
    """Replace large expanded stamp index tensors with nearest-neighbor Resize.

    The baseline already implements the verified task158 rule:
    source 3x3 motif -> orientable fill mask -> scale 1/2/3 marker pair stamping.
    Its scale-2 and scale-3 stamp weights are produced by Gather with large
    static index tensors. Since those weights are just nearest-neighbor
    upscales of the scale-1 orientable stamp mask, Resize preserves behavior
    while removing 144 + 324 index parameters.
    """
    graph = model.graph

    keep_inits = [init for init in graph.initializer if init.name not in {"stamp_idx2", "stamp_idx3"}]
    del graph.initializer[:]
    graph.initializer.extend(keep_inits)
    graph.initializer.extend(
        [
            numpy_helper.from_array(np.array([1, 4, 6, 6], dtype=np.int64), name="stamp_size2"),
            numpy_helper.from_array(np.array([1, 4, 9, 9], dtype=np.int64), name="stamp_size3"),
        ]
    )

    new_nodes = []
    for node in graph.node:
        if list(node.output) == ["stamp_w2"] and node.op_type == "Gather":
            new_nodes.append(
                helper.make_node(
                    "Resize",
                    ["stamp_w1", "", "", "stamp_size2"],
                    ["stamp_w2"],
                    name="stamp_w2",
                    mode="nearest",
                    coordinate_transformation_mode="asymmetric",
                    nearest_mode="floor",
                )
            )
            continue
        if list(node.output) == ["stamp_w3"] and node.op_type == "Gather":
            new_nodes.append(
                helper.make_node(
                    "Resize",
                    ["stamp_w1", "", "", "stamp_size3"],
                    ["stamp_w3"],
                    name="stamp_w3",
                    mode="nearest",
                    coordinate_transformation_mode="asymmetric",
                    nearest_mode="floor",
                )
            )
            continue
        new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)
    return model


def build(baseline: Path, output: Path) -> Path:
    if not baseline.exists():
        raise FileNotFoundError(f"baseline ONNX not found: {baseline}")
    output.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(baseline))
    model = replace_scaled_stamp_gathers(model)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(output))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    out = build(args.baseline, args.output)
    print(out)


if __name__ == "__main__":
    main()
