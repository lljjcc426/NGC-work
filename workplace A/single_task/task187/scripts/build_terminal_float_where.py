from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def replace_initializer(model: onnx.ModelProto, name: str, dtype: np.dtype) -> None:
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name != name:
            continue
        value = numpy_helper.to_array(initializer).astype(dtype)
        model.graph.initializer[index].CopyFrom(numpy_helper.from_array(value, name=name))
        return
    raise RuntimeError(f"initializer not found: {name}")


def build(parent_path: Path, output_path: Path) -> None:
    model = onnx.load(parent_path)
    nodes = list(model.graph.node)
    cast_index = next(
        index
        for index, node in enumerate(nodes)
        if node.op_type == "Cast" and list(node.input) == ["safe_name_118"]
    )
    cast = nodes[cast_index]
    if list(cast.output) != ["safe_name_119"]:
        raise RuntimeError("unexpected terminal Cast output")
    where = next(node for node in nodes if node.op_type == "Where" and node.name == "direct_fill_color_two")
    if where.input[2] != "safe_name_119":
        raise RuntimeError("terminal Where no longer consumes the expected Cast")
    where.input[2] = "safe_name_118"
    replace_initializer(model, "fill_color_two", np.float32)
    replace_initializer(model, "safe_name_22", np.float32)
    del model.graph.node[cast_index]
    # The parent stores inferred uint8 annotations for the old Cast path.
    # Re-infer them after changing the terminal path to float32.
    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    reloaded = onnx.load(output_path)
    onnx.checker.check_model(reloaded, full_check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Keep task187 terminal decode in float and remove its full Cast tensor.")
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.parent, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
