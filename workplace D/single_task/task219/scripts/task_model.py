from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import onnx
from onnx import TensorProto, helper


CAST_OUTPUTS_TO_REMOVE = {
    "con_62_u8b",
    "con_67_u8b",
    "mul_48_u8b",
    "con_72_u8b",
}

FLOAT_BANK_INPUTS = {
    "con_62_u8b": "con_62",
    "con_67_u8b": "con_67",
    "mul_48_u8b": "mul_48",
    "con_72_u8b": "con_72",
}

SCORE_BLOCKS = {
    "mul_192": (
        "gat_189",
        "con_210",
        {
            "mul_192", "red_193", "mul_196", "red_197", "mul_200", "red_201",
            "mul_204", "red_205", "uns_206", "uns_207", "uns_208", "uns_209", "con_210",
        },
    ),
    "mul_268": (
        "gat_265",
        "con_286",
        {
            "mul_268", "red_269", "mul_272", "red_273", "mul_276", "red_277",
            "mul_280", "red_281", "uns_282", "uns_283", "uns_284", "uns_285", "con_286",
        },
    ),
    "mul_344": (
        "gat_316",
        "con_362",
        {
            "mul_344", "red_345", "mul_348", "red_349", "mul_352", "red_353",
            "mul_356", "red_357", "uns_358", "uns_359", "uns_360", "uns_361", "con_362",
        },
    ),
}

SELECTED_PATTERNS = {"squ_215", "squ_291", "squ_367", "squ_519"}


def build(source: Path, output: Path) -> Path:
    model = deepcopy(onnx.load(source))
    all_score_outputs = set().union(*(spec[2] for spec in SCORE_BLOCKS.values()))
    nodes = []

    for original in model.graph.node:
        output_name = original.output[0] if original.output else ""

        if output_name in CAST_OUTPUTS_TO_REMOVE:
            continue

        if output_name in SCORE_BLOCKS:
            candidate_name, final_name, _ = SCORE_BLOCKS[output_name]
            nodes.append(
                helper.make_node(
                    "Einsum",
                    ["con_78_i8", candidate_name],
                    [final_name],
                    equation="abc,bc->a",
                    name=f"score_{candidate_name}",
                )
            )
            continue
        if output_name in all_score_outputs:
            continue

        node = deepcopy(original)
        for index, name in enumerate(node.input):
            if name in FLOAT_BANK_INPUTS:
                node.input[index] = FLOAT_BANK_INPUTS[name]

        if output_name in SELECTED_PATTERNS and node.op_type == "Squeeze":
            float_name = f"{output_name}_f16"
            node.output[0] = float_name
            nodes.append(node)
            nodes.append(
                helper.make_node(
                    "Cast",
                    [float_name],
                    [output_name],
                    to=TensorProto.UINT8,
                    name=f"cast_{output_name}_u8",
                )
            )
            continue

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
