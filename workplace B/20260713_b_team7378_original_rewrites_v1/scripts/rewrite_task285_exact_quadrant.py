from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

import build_blend
import optimize_equivalent as oe


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = (
    ROOT
    / "public_probe_variants"
    / "team_pending_b291_b368_b369_plus_b_tail5_20260713"
    / "submission"
    / "task285.onnx"
)
DEFAULT_OUT = (
    ROOT
    / "reconstruction_candidates"
    / "b_task285_exact_quadrant_v1"
    / "task285.onnx"
)


def _replace_initializer(
    model: onnx.ModelProto,
    name: str,
    value: np.ndarray,
) -> None:
    tensor = numpy_helper.from_array(value, name)
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name == name:
            model.graph.initializer[index].CopyFrom(tensor)
            return
    model.graph.initializer.append(tensor)


def _node(op_type: str, inputs: list[str], output: str, **attrs) -> onnx.NodeProto:
    return helper.make_node(op_type, inputs, [output], name=output, **attrs)


def rewrite(base: onnx.ModelProto) -> onnx.ModelProto:
    model = onnx.ModelProto.FromString(base.SerializeToString())

    # ARC-GEN permits three 8-cell sprites and three visible remote anchors for
    # each sprite: 3 * (8 + 3) = 33 active cells.
    _replace_initializer(model, "k32", np.array([33], dtype=np.int64))
    _replace_initializer(model, "shape_3x25", np.array([3, 25], dtype=np.int64))
    _replace_initializer(model, "shape_3155", np.array([3, 1, 5, 5], dtype=np.int64))
    _replace_initializer(model, "two_i32", np.array(2, dtype=np.int32))
    _replace_initializer(model, "padamt", np.array([0, 31], dtype=np.int64))
    _replace_initializer(
        model,
        "Wc",
        np.arange(1, 11, dtype=np.float32).reshape(1, 10, 1, 1),
    )
    _replace_initializer(model, "signed_color_bias", np.array([-1.0], dtype=np.float32))
    _replace_initializer(
        model,
        "ar10",
        np.arange(10, dtype=np.int8).reshape(1, 10, 1, 1),
    )
    _replace_initializer(model, "zero_i8", np.array(0, dtype=np.int8))
    _replace_initializer(model, "neg_one_f16", np.array(-1, dtype=np.float16))
    _replace_initializer(
        model,
        "orthogonal_offsets",
        np.array([[-1], [1], [-30], [30]], dtype=np.int32),
    )
    _replace_initializer(
        model,
        "diagonal_offsets",
        np.array([[-31], [-29], [29], [31]], dtype=np.int32),
    )

    ii, jj = np.meshgrid(
        np.arange(5, dtype=np.int32),
        np.arange(5, dtype=np.int32),
        indexing="ij",
    )
    # The row is selected by the source quadrant. The index is
    # 2 * (marker_row_direction < 0) + (marker_col_direction < 0).
    quadrant_offsets = np.stack(
        [
            -30 * ii - jj,
            -30 * ii + jj,
            30 * ii - jj,
            30 * ii + jj,
        ]
    ).reshape(4, 25)
    _replace_initializer(model, "quadrant_offsets", quadrant_offsets.astype(np.int32))
    _replace_initializer(model, "local_i", ii.reshape(25).astype(np.float16))
    _replace_initializer(model, "local_j", jj.reshape(25).astype(np.float16))
    seed = np.zeros((1, 1, 5, 5), dtype=np.uint8)
    seed[0, 0, 0, 0] = 1
    _replace_initializer(model, "quadrant_seed", seed)

    prefix: list[onnx.NodeProto] = []
    for node in model.graph.node:
        prefix.append(onnx.NodeProto.FromString(node.SerializeToString()))
        if "c" in node.output:
            break
    else:
        raise RuntimeError("task285 color prefix ending at tensor 'c' was not found")

    for node in prefix:
        if "cf" in node.output:
            node.input.append("signed_color_bias")
        if node.op_type == "Cast" and any(output in {"c2d", "c"} for output in node.output):
            for attr in node.attribute:
                if attr.name == "to":
                    attr.i = TensorProto.INT8

    nodes = prefix

    # Four-neighbor tests are sufficient to identify each shown source corner.
    # The full eight-neighbor table is only needed after the three anchors have
    # been selected, reducing the largest sparse intermediate by half.
    nodes.extend(
        [
            _node("Add", ["t", "orthogonal_offsets"], "exact_neighbor_idx"),
            _node("Gather", ["gp", "exact_neighbor_idx"], "exact_neighbor_color"),
            _node("Equal", ["exact_neighbor_color", "c"], "exact_same_bool"),
            _node("Add", ["t", "diagonal_offsets"], "exact_diagonal_idx"),
            _node("Gather", ["gp", "exact_diagonal_idx"], "exact_diagonal_color"),
            _node("Equal", ["exact_diagonal_color", "c"], "exact_diagonal_same_bool"),
            _node("Or", ["exact_same_bool", "exact_diagonal_same_bool"], "exact_any_same_bool"),
            _node("Cast", ["exact_any_same_bool"], "exact_any_same_i8", to=TensorProto.INT8),
            _node("ReduceMax", ["exact_any_same_i8", "s0"], "exact_has_same", keepdims=0),
            _node("Greater", ["exact_neighbor_color", "zero_i8"], "exact_neighbor_nonzero"),
            _node("Xor", ["exact_neighbor_nonzero", "exact_same_bool"], "exact_other_bool"),
            _node("Cast", ["exact_other_bool"], "exact_other_u8", to=TensorProto.INT8),
            _node("ReduceMax", ["exact_other_u8", "s0"], "exact_has_other", keepdims=0),
            _node("Mul", ["exact_has_same", "exact_has_other"], "exact_anchor_score_0"),
            _node("Mul", ["exact_anchor_score_0", "c"], "exact_anchor_score"),
            _node("Cast", ["exact_anchor_score"], "exact_anchor_score_f16", to=TensorProto.FLOAT16),
            helper.make_node(
                "TopK",
                ["exact_anchor_score_f16", "k3"],
                ["sv", "si"],
                name="sv",
                axis=0,
                largest=1,
                sorted=0,
            ),
            _node("Greater", ["sv", "f0"], "avalid"),
            _node("Unsqueeze", ["si", "axs1"], "si2"),
            _node("Gather", ["t", "si2"], "a"),
            _node("Gather", ["c", "si2"], "acol"),
            _node("Transpose", ["a"], "exact_anchor_row", perm=[1, 0]),
            _node("Add", ["OFF8", "exact_anchor_row"], "exact_orient_idx"),
            _node("Gather", ["gp", "exact_orient_idx"], "exact_orient_color"),
            _node("Transpose", ["acol"], "exact_anchor_color_row", perm=[1, 0]),
            _node("Equal", ["exact_orient_color", "exact_anchor_color_row"], "exact_orient_same"),
            _node("Greater", ["exact_orient_color", "zero_i8"], "exact_orient_nonzero"),
            _node("Xor", ["exact_orient_nonzero", "exact_orient_same"], "exact_orient_other"),
            _node("Transpose", ["exact_orient_other"], "exact_orient_other_t", perm=[1, 0]),
            _node("Cast", ["exact_orient_other_t"], "seedT", to=TensorProto.FLOAT16),
            _node("MatMul", ["seedT", "MT"], "E"),
            helper.make_node(
                "Split",
                ["E"],
                ["e0", "f0v", "dxv", "dyv"],
                name="e0",
                axis=1,
                num_outputs=4,
            ),
            _node("Equal", ["e0", "f0"], "ze0"),
            _node("Where", ["ze0", "dxv", "e0"], "e"),
            _node("Equal", ["f0v", "f0"], "zf0"),
            _node("Where", ["zf0", "dyv", "f0v"], "f"),
        ]
    )

    # Resolve the exact source quadrant from the other visible center pixels.
    nodes.extend(
        [
            _node("Less", ["e", "f0"], "exact_e_neg"),
            _node("Less", ["f", "f0"], "exact_f_neg"),
            _node("Cast", ["exact_e_neg"], "exact_e_neg_i", to=TensorProto.INT32),
            _node("Cast", ["exact_f_neg"], "exact_f_neg_i", to=TensorProto.INT32),
            _node("Mul", ["exact_f_neg_i", "two_i32"], "exact_f_pair"),
            _node("Add", ["exact_f_pair", "exact_e_neg_i"], "exact_quadrant_2d"),
            _node("Reshape", ["exact_quadrant_2d", "k3"], "exact_quadrant"),
            _node("Gather", ["quadrant_offsets", "exact_quadrant"], "exact_offsets"),
            _node("Add", ["exact_offsets", "a"], "exact_window_idx"),
            _node("Gather", ["gp", "exact_window_idx"], "exact_window_color"),
            _node("Equal", ["exact_window_color", "acol"], "exact_member_bool"),
            _node("Cast", ["exact_member_bool"], "exact_member_u8", to=TensorProto.UINT8),
            _node("Reshape", ["exact_member_u8", "shape_3155"], "exact_member_grid"),
            _node("Mul", ["exact_member_grid", "quadrant_seed"], "exact_flood_0"),
        ]
    )

    # Exhaustive enumeration of the generator's connected subsets shows that
    # six eight-neighbor expansions cover every fresh eight-cell creature. The
    # one legacy nine-cell training creature also fits this bound.
    previous = "exact_flood_0"
    for iteration in range(1, 7):
        pooled = f"exact_pool_{iteration}"
        flooded = f"exact_flood_{iteration}"
        nodes.append(
            _node(
                "MaxPool",
                [previous],
                pooled,
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
            )
        )
        nodes.append(_node("Mul", [pooled, "exact_member_grid"], flooded))
        previous = flooded

    nodes.extend(
        [
            _node("Reshape", [previous, "shape_3x25"], "exact_member_flat"),
            _node("Cast", ["exact_member_flat"], "exact_member_f16", to=TensorProto.FLOAT16),
            helper.make_node(
                "TopK",
                ["exact_member_f16", "k9"],
                ["exact_member_value", "exact_member_index"],
                name="exact_member_value",
                axis=1,
                largest=1,
                sorted=0,
            ),
            _node("Greater", ["exact_member_value", "f0"], "exact_member_valid"),
            _node("Unsqueeze", ["avalid", "axs1"], "exact_anchor_valid"),
            _node("And", ["exact_member_valid", "exact_anchor_valid"], "exact_valid"),
            _node("Gather", ["local_i", "exact_member_index"], "exact_i"),
            _node("Gather", ["local_j", "exact_member_index"], "exact_j"),
            _node("Mul", ["f", "exact_i"], "exact_fi"),
            _node("Mul", ["exact_fi", "f30"], "exact_fi30"),
            _node("Mul", ["e", "exact_j"], "exact_ej"),
            _node("Neg", ["exact_fi30"], "exact_neg_fi30"),
            _node("Add", ["exact_neg_fi30", "exact_ej"], "exact_h_delta_0"),
            _node("Add", ["exact_h_delta_0", "e"], "exact_h_delta"),
            _node("Sub", ["exact_fi30", "exact_ej"], "exact_v_delta_0"),
            _node("Mul", ["f", "f30"], "exact_f30"),
            _node("Add", ["exact_v_delta_0", "exact_f30"], "exact_v_delta"),
            _node("Add", ["exact_fi30", "exact_ej"], "exact_d_delta_0"),
            _node("Add", ["exact_d_delta_0", "exact_f30"], "exact_d_delta_1"),
            _node("Add", ["exact_d_delta_1", "e"], "exact_d_delta"),
            _node("Cast", ["a"], "exact_anchor_f16", to=TensorProto.FLOAT16),
            _node("Add", ["exact_anchor_f16", "exact_h_delta"], "exact_h_idx"),
            _node("Add", ["exact_anchor_f16", "exact_v_delta"], "exact_v_idx"),
            _node("Add", ["exact_anchor_f16", "exact_d_delta"], "exact_d_idx"),
            _node("Unsqueeze", ["exact_h_idx", "axs2"], "exact_h_idx_3"),
            _node("Unsqueeze", ["exact_v_idx", "axs2"], "exact_v_idx_3"),
            _node("Unsqueeze", ["exact_d_idx", "axs2"], "exact_d_idx_3"),
            helper.make_node(
                "Concat",
                ["exact_h_idx_3", "exact_v_idx_3", "exact_d_idx_3"],
                ["exact_target_idx_3"],
                name="exact_target_idx_3",
                axis=2,
            ),
            _node("Reshape", ["exact_target_idx_3", "sh81"], "exact_target_idx_f16"),
            _node("Cast", ["exact_target_idx_f16"], "exact_target_idx", to=TensorProto.INT32),
            _node("Cast", ["e"], "exact_e_i32", to=TensorProto.INT32),
            _node("Cast", ["exact_f30"], "exact_f30_i32", to=TensorProto.INT32),
            _node("Add", ["a", "exact_e_i32"], "exact_h_anchor"),
            _node("Add", ["a", "exact_f30_i32"], "exact_v_anchor"),
            _node("Add", ["exact_v_anchor", "exact_e_i32"], "exact_d_anchor"),
            _node("Gather", ["gp", "exact_h_anchor"], "exact_h_color"),
            _node("Gather", ["gp", "exact_v_anchor"], "exact_v_color"),
            _node("Gather", ["gp", "exact_d_anchor"], "exact_d_color"),
            _node("Unsqueeze", ["exact_h_color", "axs2"], "exact_h_color_3"),
            _node("Unsqueeze", ["exact_v_color", "axs2"], "exact_v_color_3"),
            _node("Unsqueeze", ["exact_d_color", "axs2"], "exact_d_color_3"),
            helper.make_node(
                "Concat",
                ["exact_h_color_3", "exact_v_color_3", "exact_d_color_3"],
                ["exact_target_colors"],
                name="exact_target_colors",
                axis=2,
            ),
            _node("Unsqueeze", ["exact_valid", "axs2"], "exact_valid_3"),
            _node("Cast", ["exact_target_colors"], "exact_target_colors_f16", to=TensorProto.FLOAT16),
            _node("Where", ["exact_valid_3", "exact_target_colors_f16", "neg_one_f16"], "exact_updates_f16"),
            _node("Cast", ["exact_updates_f16"], "exact_updates_3", to=TensorProto.INT8),
            _node("Reshape", ["exact_updates_3", "sh81"], "exact_updates"),
            helper.make_node(
                "ScatterElements",
                ["g", "exact_target_idx", "exact_updates"],
                ["newg"],
                name="newg",
                axis=0,
                reduction="max",
            ),
            _node("Reshape", ["newg", "sh2d"], "out2d"),
        ]
    )

    # The signed scalar map already uses -1 outside the generated square, so no
    # separate boundary sentinel field is needed.
    nodes.append(_node("Equal", ["out2d", "ar10"], "output"))

    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.value_info[:]
    oe.prune_dead(model)
    oe.prune_initializers(model)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model


def stress_test(path: Path, examples: int, seed: int) -> tuple[int, int]:
    arc_gen_root = ROOT / "external" / "ARC-GEN"
    sys.path.insert(0, str(arc_gen_root))
    from tasks import task_b775ac94  # type: ignore  # noqa: PLC0415

    session = ort.InferenceSession(path.read_bytes(), providers=["CPUExecutionProvider"])
    random.seed(seed)
    right = 0
    for index in range(examples):
        example = task_b775ac94.generate()
        pair = build_blend.convert_to_numpy(example)
        if pair is None:
            raise RuntimeError("generator produced an unsupported grid")
        actual = session.run(["output"], {"input": pair["input"]})[0]
        if not np.array_equal(actual, pair["output"]):
            return right, index + 1
        right += 1
    return right, examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stress", type=int, default=0)
    parser.add_argument("--seed", type=int, default=285_20260713)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model = rewrite(onnx.load(args.base))
    onnx.save(model, args.out)
    score = build_blend.validate_and_score((285, "task285_exact_quadrant", str(args.out)))
    result: dict[str, object] = {"score": score}
    if args.stress:
        right, total = stress_test(args.out, args.stress, args.seed)
        result["stress"] = {"right": right, "total": total, "seed": args.seed}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
