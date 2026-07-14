from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import build_blend  # noqa: E402
import optimize_equivalent as oe  # noqa: E402


DEFAULT_BASE = (
    ROOT
    / "reconstruction_candidates"
    / "b_task076_f16_coordinates_v1"
    / "task076.onnx"
)
DEFAULT_OUT = (
    ROOT
    / "reconstruction_candidates"
    / "b_task076_u8_geometry_v2"
    / "task076.onnx"
)


INDEX_PATHS = {
    "target_topk_indices": ("target_topk_f16", "target_topk_row_div", "target_rows_abs", "target_cols_abs"),
    "source_hidden_topk_indices": (
        "source_hidden_topk_f16",
        "source_hidden_topk_row_div",
        "source_hidden_rows_abs",
        "source_hidden_cols_abs",
    ),
    "source_visible_topk_indices": (
        "source_visible_topk_f16",
        "source_visible_topk_row_div",
        "source_visible_rows_abs",
        "source_visible_cols_abs",
    ),
}


def rewrite(model: onnx.ModelProto) -> onnx.ModelProto:
    current = onnx.ModelProto.FromString(model.SerializeToString())
    outputs = {out for node in current.graph.node for out in node.output}
    required = {name for names in INDEX_PATHS.values() for name in names}
    required.update({"vrows_stack_u8", "vcols_stack_u8", "source_red_row_abs", "source_red_col_abs"})
    if not required.issubset(outputs):
        raise RuntimeError(f"task076 graph no longer matches u8 geometry rewrite: {sorted(required - outputs)}")

    coordinate_remove = {name for names in INDEX_PATHS.values() for name in names}
    coordinate_insert: dict[str, list[onnx.NodeProto]] = {}
    for indices, (index_f16, _, rows, cols) in INDEX_PATHS.items():
        index_u8 = index_f16.removesuffix("_f16") + "_u8"
        coordinate_insert[index_f16] = [
            helper.make_node(
                "Cast",
                [indices],
                [index_u8],
                name=f"{index_u8}_cast",
                to=TensorProto.UINT8,
            ),
            helper.make_node("Div", [index_u8, "geom_15_u8"], [rows], name=f"{rows}_u8"),
            helper.make_node("Mod", [index_u8, "geom_15_u8"], [cols], name=f"{cols}_u8"),
        ]

    remove_outputs = coordinate_remove | {"target_rows_3d", "target_cols_3d", "vrows_stack_bf", "vcols_stack_bf"}
    replacements = {
        "target_rows_3d": "target_rows_abs",
        "target_cols_3d": "target_cols_abs",
    }
    rewritten: list[onnx.NodeProto] = []
    inserted: set[str] = set()

    for node in current.graph.node:
        original_output = node.output[0] if node.output else ""
        if original_output in coordinate_insert:
            rewritten.extend(coordinate_insert[original_output])
            inserted.add(original_output)
            continue
        if original_output in remove_outputs:
            continue

        updated = onnx.NodeProto.FromString(node.SerializeToString())
        for index, name in enumerate(updated.input):
            updated.input[index] = replacements.get(name, name)

        if original_output == "target_rows_2d":
            rewritten.append(
                helper.make_node(
                    "Cast",
                    ["target_rows_abs"],
                    ["target_rows_hidden_f16"],
                    name="target_rows_hidden_f16",
                    to=TensorProto.FLOAT16,
                )
            )
            updated.input[0] = "target_rows_hidden_f16"
        elif original_output == "source_red_row_abs":
            rewritten.append(updated)
            rewritten.extend(
                [
                    helper.make_node(
                        "Cast",
                        ["source_hidden_rows_abs"],
                        ["source_hidden_rows_f16"],
                        name="source_hidden_rows_f16",
                        to=TensorProto.FLOAT16,
                    ),
                    helper.make_node(
                        "Cast",
                        ["source_red_row_abs"],
                        ["source_red_row_f16"],
                        name="source_red_row_f16",
                        to=TensorProto.FLOAT16,
                    ),
                ]
            )
            continue
        elif original_output == "hidden_dr":
            updated.input[0] = "source_hidden_rows_f16"
            updated.input[1] = "source_red_row_f16"
        elif original_output == "source_red_col_abs":
            rewritten.append(updated)
            rewritten.extend(
                [
                    helper.make_node(
                        "Cast",
                        ["source_hidden_cols_abs"],
                        ["source_hidden_cols_f16"],
                        name="source_hidden_cols_f16",
                        to=TensorProto.FLOAT16,
                    ),
                    helper.make_node(
                        "Cast",
                        ["source_red_col_abs"],
                        ["source_red_col_f16"],
                        name="source_red_col_f16",
                        to=TensorProto.FLOAT16,
                    ),
                ]
            )
            continue
        elif original_output == "hidden_dc":
            updated.input[0] = "source_hidden_cols_f16"
            updated.input[1] = "source_red_col_f16"
        elif original_output in {"visible_neg_dc", "visible_neg_dr"}:
            updated.input[0] = "zero_u8"
        elif original_output == "vrows_stack_u8":
            rewritten.append(
                helper.make_node(
                    "Add",
                    ["visible_rows_stack", "geom_b_u8"],
                    ["vrows_stack_u8"],
                    name="vrows_stack_u8",
                )
            )
            continue
        elif original_output == "vcols_stack_u8":
            rewritten.append(
                helper.make_node(
                    "Add",
                    ["visible_cols_stack", "geom_b_u8"],
                    ["vcols_stack_u8"],
                    name="vcols_stack_u8",
                )
            )
            continue
        elif original_output == "target_cols_2d":
            rewritten.append(
                helper.make_node(
                    "Cast",
                    ["target_cols_abs"],
                    ["target_cols_hidden_f16"],
                    name="target_cols_hidden_f16",
                    to=TensorProto.FLOAT16,
                )
            )
            updated.input[0] = "target_cols_hidden_f16"

        rewritten.append(updated)

    if inserted != set(coordinate_insert):
        raise RuntimeError(f"failed to replace coordinate paths: {sorted(set(coordinate_insert) - inserted)}")

    del current.graph.node[:]
    current.graph.node.extend(rewritten)
    del current.graph.value_info[:]
    oe.prune_dead(current)
    oe.prune_initializers(current)
    onnx.checker.check_model(current, full_check=True)
    return current


def _equivalence(base: Path, candidate: Path) -> dict[str, int]:
    base_session = ort.InferenceSession(str(base), providers=["CPUExecutionProvider"])
    candidate_session = ort.InferenceSession(str(candidate), providers=["CPUExecutionProvider"])
    examples = json.loads((ROOT / "data" / "competition" / "task076.json").read_text())
    checked = 0
    for split in ("train", "test", "arc-gen"):
        for example in examples.get(split, []):
            pair = build_blend.convert_to_numpy(example)
            if pair is None:
                continue
            expected = base_session.run(["output"], {"input": pair["input"]})[0]
            actual = candidate_session.run(["output"], {"input": pair["input"]})[0]
            if not np.array_equal(expected, actual):
                raise RuntimeError(f"equivalence failed in {split} example {checked}")
            checked += 1
    return {"checked": checked, "matched": checked}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(rewrite(onnx.load(args.base)), args.out)
    result = {
        "task": 76,
        "method": "uint8 modular visible geometry with float16 hidden coordinates",
        "equivalence": _equivalence(args.base, args.out),
        "score": build_blend.validate_and_score((76, "task076_u8_geometry", str(args.out))),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
