from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def build(parent_path: Path, output_path: Path) -> None:
    model = onnx.load(parent_path)
    nodes = list(model.graph.node)
    terminal_start = next(index for index, node in enumerate(nodes) if node.output and node.output[0] == "safe_name_118")
    if [node.op_type for node in nodes[terminal_start:]] != ["Conv", "Cast", "Where", "Pad", "Equal"]:
        raise RuntimeError("unexpected task187 terminal chain")

    fill = np.zeros((1, 10, 1, 1), dtype=np.bool_)
    fill[:, 2, :, :] = True
    fill_name = "fill_color_two_onehot"
    model.graph.initializer.append(numpy_helper.from_array(fill, name=fill_name))
    pad_mask = helper.make_node(
        "Pad",
        ["safe_name_116", "v147_pads_90_29", "", "v147_axes_90_30"],
        ["fill_mask_30"],
        name="pad_fill_mask",
        mode="constant",
    )
    direct = helper.make_node(
        "Where",
        ["fill_mask_30", fill_name, "input"],
        ["output"],
        name="direct_onehot_fill",
    )
    del model.graph.node[terminal_start:]
    model.graph.node.extend([pad_mask, direct])

    used = {name for node in model.graph.node for name in node.input if name}
    kept = [initializer for initializer in model.graph.initializer if initializer.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    reloaded = onnx.load(output_path)
    onnx.checker.check_model(reloaded, full_check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Replace task187 scalar decode/re-encode with a direct one-hot overlay.")
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.parent, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
