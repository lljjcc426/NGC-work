from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


REPLACED_OUTPUTS = {
    "line",
    "line_b",
    "line30",
    "output",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = onnx.load(args.source)
    kept = [
        node
        for node in model.graph.node
        if not any(output in REPLACED_OUTPUTS for output in node.output)
    ]

    kept.extend(
        [
            helper.make_node(
                "Equal",
                ["hline", "vline"],
                ["no_line"],
                name="complement_equal",
            ),
            helper.make_node(
                "Pad",
                ["no_line", "pads_hw", "btrue", "ax23"],
                ["no_line30"],
                name="pad_no_line",
            ),
            helper.make_node(
                "Where",
                ["no_line30", "input", "marker_ohf"],
                ["output"],
                name="paint_line_by_complement",
            ),
        ]
    )

    del model.graph.node[:]
    model.graph.node.extend(kept)
    model.graph.initializer.append(
        numpy_helper.from_array(np.asarray(True, dtype=np.bool_), name="btrue")
    )
    onnx.checker.check_model(model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
