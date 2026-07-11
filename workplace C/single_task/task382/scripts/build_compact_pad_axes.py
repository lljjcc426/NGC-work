from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


PAD_SPECS = {
    "h_pos_pad4": (np.array([4, 0], dtype=np.int64), "col_axes"),
    "h_neg_pad4": (np.array([0, 4], dtype=np.int64), "col_axes"),
    "v_pos_pad4": (np.array([4, 0], dtype=np.int64), "row_axes"),
    "v_neg_pad4": (np.array([0, 4], dtype=np.int64), "row_axes"),
    "color_index_30": (np.array([0, 0, 10, 10], dtype=np.int64), "pad_hw_axes"),
}


def build(source: Path, output: Path) -> Path:
    model = deepcopy(onnx.load(source))
    model.opset_import[0].version = 18
    initializers = {item.name: item for item in model.graph.initializer}
    model.graph.initializer.append(
        numpy_helper.from_array(np.array([2, 3], dtype=np.int64), name="pad_hw_axes")
    )

    replaced_pad_initializers: set[str] = set()
    for node in model.graph.node:
        if node.op_type == "Pad" and node.output[0] in PAD_SPECS:
            compact_pads, axes_name = PAD_SPECS[node.output[0]]
            old_pads_name = node.input[1]
            new_pads_name = f"{old_pads_name}_compact"
            model.graph.initializer.append(numpy_helper.from_array(compact_pads, name=new_pads_name))
            node.input[1] = new_pads_name
            while len(node.input) < 3:
                node.input.append("")
            node.input.append(axes_name)
            replaced_pad_initializers.add(old_pads_name)
        elif node.op_type == "Squeeze":
            # Every dimension omitted by the original axes is non-singleton;
            # omitting axes therefore produces the same 1-D/scalar outputs.
            del node.attribute[:]
        elif node.op_type == "ReduceSum" and node.output[0] == "x8_row_count":
            axes_attributes = [attr for attr in node.attribute if attr.name == "axes"]
            if len(axes_attributes) != 1:
                raise RuntimeError("unexpected task382 ReduceSum")
            kept = [attr for attr in node.attribute if attr.name != "axes"]
            del node.attribute[:]
            node.attribute.extend(kept)
            node.input.append("row_axes")

    kept = [item for item in model.graph.initializer if item.name not in replaced_pad_initializers]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)

    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.source, args.output))


if __name__ == "__main__":
    main()
