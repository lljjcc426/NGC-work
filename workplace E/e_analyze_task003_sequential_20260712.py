#!/usr/bin/env python
"""Inspect task003 from the teammate-high sequential-loop baseline."""
from __future__ import annotations

import json
import pathlib
import sys
import zipfile


sys.path.extend(
    [
        r"C:\ProgramData\anaconda3\Lib\site-packages",
        r"C:\Users\cc\AppData\Roaming\Python\Python311\site-packages",
    ]
)

import onnx  # noqa: E402
from onnx import numpy_helper  # noqa: E402


BASE_ZIP = pathlib.Path(r"F:\kaggle\submission (1).zip")
DATA_JSON = pathlib.Path(r"F:\kaggle\neurogolf-2026\data\task003.json")


def main() -> int:
    with zipfile.ZipFile(BASE_ZIP) as archive:
        payload = archive.read("task003.onnx")
    model = onnx.load_from_string(payload)
    print(f"model_bytes={len(payload)} nodes={len(model.graph.node)}")
    print(f"opsets={[(item.domain, item.version) for item in model.opset_import]}")
    for index, node in enumerate(model.graph.node):
        attrs = {
            attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute
        }
        print(
            index,
            node.op_type,
            list(node.input),
            "->",
            list(node.output),
            attrs,
        )
    for initializer in model.graph.initializer:
        value = numpy_helper.to_array(initializer)
        print(
            "initializer",
            initializer.name,
            list(value.shape),
            value.dtype,
            value.tolist(),
        )

    examples = json.loads(DATA_JSON.read_text(encoding="utf-8"))
    print("example_counts", {name: len(rows) for name, rows in examples.items()})
    for index, example in enumerate(examples["train"]):
        input_grid = example["input"]
        output_grid = example["output"]
        print(
            "train",
            index,
            "input_shape",
            [len(input_grid), len(input_grid[0])],
            "output_shape",
            [len(output_grid), len(output_grid[0])],
        )
        print("input", input_grid)
        print("output", output_grid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
