from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import onnx


REMOVE_OUTPUTS = {
    "safe_name_101",
    "safe_name_110",
    "safe_name_119",
    "safe_name_137",
    "safe_name_146",
    "safe_name_155",
    "safe_name_164",
}

REWIRE = {
    "safe_name_102": ("safe_name_99", "safe_name_100"),
    "safe_name_111": ("safe_name_108", "safe_name_109"),
    "safe_name_120": ("safe_name_117", "safe_name_118"),
    "safe_name_138": ("safe_name_135", "safe_name_136"),
    "safe_name_147": ("safe_name_144", "safe_name_145"),
    "safe_name_156": ("safe_name_153", "safe_name_154"),
    "safe_name_165": ("safe_name_162", "safe_name_163"),
}


def build(source: Path, output: Path) -> Path:
    model = deepcopy(onnx.load(source))
    nodes = []
    for original in model.graph.node:
        output_name = original.output[0] if original.output else ""
        if output_name in REMOVE_OUTPUTS:
            continue
        node = deepcopy(original)
        if output_name in REWIRE:
            del node.input[:]
            node.input.extend(REWIRE[output_name])
        nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)

    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
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
