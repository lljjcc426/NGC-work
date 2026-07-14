from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def build(parent_path: Path, output_path: Path) -> None:
    model = onnx.load(parent_path)
    nodes = list(model.graph.node)
    if [node.op_type for node in nodes[:3]] != ["Conv", "Cast", "Mod"]:
        raise RuntimeError("unexpected task364 prefix")
    if list(nodes[2].input) != ["Fu", "twou"] or list(nodes[2].output) != ["Gu"]:
        raise RuntimeError("unexpected task364 modulo path")
    weight = next(initializer for initializer in model.graph.initializer if initializer.name == "Wcol2")
    values = numpy_helper.to_array(weight).copy()
    if values.reshape(-1).tolist() != [2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]:
        raise RuntimeError("unexpected task364 color projection")
    values.reshape(-1)[0] = 0.0
    weight.CopyFrom(numpy_helper.from_array(values, name="Wcol2"))
    nodes[1].output[0] = "Gu"
    final_add = next(node for node in nodes if list(node.output) == ["v1"])
    if list(final_add.input) != ["SE5", "Fu"]:
        raise RuntimeError("unexpected task364 terminal Add")
    final_add.input[1] = "Gu"
    terminal_pad = next(node for node in nodes if list(node.output) == ["vp"])
    if list(terminal_pad.input) != ["v1", "padv_compact", "", "padv_axes"]:
        raise RuntimeError("unexpected task364 terminal Pad")
    sentinel_name = "outside_sentinel"
    model.graph.initializer.append(numpy_helper.from_array(np.asarray(255, dtype=np.uint8), name=sentinel_name))
    terminal_pad.input[2] = sentinel_name
    channels = next(initializer for initializer in model.graph.initializer if initializer.name == "K")
    channel_values = numpy_helper.to_array(channels).copy()
    if int(channel_values.reshape(-1)[0]) != 2:
        raise RuntimeError("unexpected task364 background code")
    channel_values.reshape(-1)[0] = 0
    channels.CopyFrom(numpy_helper.from_array(channel_values, name="K"))
    del model.graph.node[2]
    used = {name for node in model.graph.node for name in node.input if name}
    kept = [initializer for initializer in model.graph.initializer if initializer.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    onnx.checker.check_model(onnx.load(output_path), full_check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fold task364's modulo-2 color projection into its Conv weights.")
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.parent, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
