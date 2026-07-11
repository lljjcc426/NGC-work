from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = (
    ROOT
    / "public_probe_variants"
    / "yusuketogashi_v176_full"
    / "submission"
    / "task018.onnx"
)
DEFAULT_OUT = (
    ROOT
    / "reconstruction_candidates"
    / "b_task018_exact_generator_v1"
    / "task018.onnx"
)


def _set_value_info(
    model: onnx.ModelProto,
    name: str,
    elem_type: int,
    shape: list[int],
) -> None:
    matches = [item for item in model.graph.value_info if item.name == name]
    if len(matches) > 1:
        raise RuntimeError(f"duplicate value_info: {name}")
    if matches:
        item = matches[0]
        item.type.tensor_type.elem_type = elem_type
        item.type.tensor_type.shape.ClearField("dim")
        for size in shape:
            item.type.tensor_type.shape.dim.add().dim_value = size
        return
    model.graph.value_info.append(helper.make_tensor_value_info(name, elem_type, shape))


def transform(model: onnx.ModelProto) -> onnx.ModelProto:
    current = onnx.ModelProto.FromString(model.SerializeToString())
    outputs = {output for node in current.graph.node for output in node.output if output}
    required = {
        "safe_name_55",
        "safe_name_63",
        "safe_name_65",
        "safe_name_66",
        "safe_name_76",
        "safe_name_79",
        "safe_name_81",
        "safe_name_90_share_cast",
        "safe_name_352",
        "safe_name_353",
        "safe_name_653",
        "safe_name_654",
        "safe_name_655",
    }
    if not required.issubset(outputs):
        missing = sorted(required - outputs)
        raise RuntimeError(f"unexpected task018 graph; missing outputs: {missing}")

    remove_outputs = {
        "safe_name_77",
        "safe_name_78",
        "safe_name_79",
        "safe_name_81",
        "safe_name_88",
        "safe_name_89",
        "safe_name_90_share_cast",
        "safe_name_353",
        "safe_name_654",
    }

    clone_anchor_nodes = [
        helper.make_node(
            "Equal",
            ["safe_name_66", "safe_name_76"],
            ["task018_clone_anchor_mask_b"],
            name="task018_clone_anchor_mask_b",
        ),
        helper.make_node(
            "Cast",
            ["task018_clone_anchor_mask_b"],
            ["task018_clone_anchor_mask_u8"],
            name="task018_clone_anchor_mask_u8",
            to=TensorProto.UINT8,
        ),
        helper.make_node(
            "TopK",
            ["task018_clone_anchor_mask_u8", "safe_name_10"],
            ["task018_clone_anchor_valid_u8", "task018_clone_anchor_idx_i64"],
            name="task018_clone_anchor_top2",
            axis=-1,
            largest=1,
            sorted=1,
        ),
        helper.make_node(
            "Cast",
            ["task018_clone_anchor_valid_u8"],
            ["task018_clone_anchor_valid_f16_raw"],
            name="task018_clone_anchor_valid_f16_raw",
            to=TensorProto.FLOAT16,
        ),
        helper.make_node(
            "Mul",
            ["task018_clone_anchor_valid_f16_raw", "safe_name_14"],
            ["safe_name_79"],
            name="task018_clone_anchor_valid_f16",
        ),
        helper.make_node(
            "Cast",
            ["task018_clone_anchor_idx_i64"],
            ["safe_name_81"],
            name="task018_clone_anchor_idx_f16",
            to=TensorProto.FLOAT16,
        ),
    ]
    full_anchor_nodes = [
        helper.make_node(
            "Cast",
            ["safe_name_87"],
            ["task018_full_anchor_mask_u8"],
            name="task018_full_anchor_mask_u8",
            to=TensorProto.UINT8,
        ),
        helper.make_node(
            "TopK",
            ["task018_full_anchor_mask_u8", "safe_name_10"],
            ["task018_full_anchor_valid_u8", "task018_full_anchor_idx_i64"],
            name="task018_full_anchor_top2",
            axis=-1,
            largest=1,
            sorted=1,
        ),
        helper.make_node(
            "Cast",
            ["task018_full_anchor_idx_i64"],
            ["safe_name_90_share_cast"],
            name="task018_full_anchor_idx_i32",
            to=TensorProto.INT32,
        ),
    ]

    inserted_clone_anchors = False
    inserted_full_anchors = False
    new_nodes: list[onnx.NodeProto] = []
    for node in current.graph.node:
        first_output = node.output[0] if node.output else ""
        if first_output in remove_outputs:
            if first_output == "safe_name_77" and not inserted_clone_anchors:
                new_nodes.extend(clone_anchor_nodes)
                inserted_clone_anchors = True
            elif first_output == "safe_name_88" and not inserted_full_anchors:
                new_nodes.extend(full_anchor_nodes)
                inserted_full_anchors = True
            continue

        copied = onnx.NodeProto.FromString(node.SerializeToString())
        if copied.op_type == "TopK" and copied.input:
            if copied.input[0] == "safe_name_353":
                copied.input[0] = "safe_name_352"
            elif copied.input[0] == "safe_name_654":
                copied.input[0] = "safe_name_653"
        if first_output == "safe_name_710":
            copied.input[1] = "safe_name_30"
        new_nodes.append(copied)

    if not (inserted_clone_anchors and inserted_full_anchors):
        raise RuntimeError("failed to insert task018 replacements")

    del current.graph.node[:]
    current.graph.node.extend(new_nodes)

    live_outputs = {
        output
        for node in current.graph.node
        for output in node.output
        if output
    }
    kept_value_info = [
        onnx.ValueInfoProto.FromString(item.SerializeToString())
        for item in current.graph.value_info
        if item.name in live_outputs
    ]
    del current.graph.value_info[:]
    current.graph.value_info.extend(kept_value_info)

    explicit_types = {
        "task018_clone_anchor_mask_b": (TensorProto.BOOL, [576]),
        "task018_clone_anchor_mask_u8": (TensorProto.UINT8, [576]),
        "task018_clone_anchor_valid_u8": (TensorProto.UINT8, [2]),
        "task018_clone_anchor_idx_i64": (TensorProto.INT64, [2]),
        "task018_clone_anchor_valid_f16_raw": (TensorProto.FLOAT16, [2]),
        "task018_full_anchor_mask_u8": (TensorProto.UINT8, [576]),
        "task018_full_anchor_valid_u8": (TensorProto.UINT8, [2]),
        "task018_full_anchor_idx_i64": (TensorProto.INT64, [2]),
        "safe_name_79": (TensorProto.FLOAT16, [2]),
        "safe_name_81": (TensorProto.FLOAT16, [2]),
        "safe_name_90_share_cast": (TensorProto.INT32, [2]),
        "safe_name_354": (TensorProto.UINT8, [12]),
        "safe_name_355": (TensorProto.INT64, [12]),
        "safe_name_655": (TensorProto.UINT8, [12]),
        "safe_name_656": (TensorProto.INT64, [12]),
    }
    for name, (elem_type, shape) in explicit_types.items():
        _set_value_info(current, name, elem_type, shape)

    onnx.checker.check_model(current, full_check=True)
    onnx.shape_inference.infer_shapes(current, strict_mode=True)
    return current


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    optimized = transform(onnx.load(args.src))
    onnx.save(optimized, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
