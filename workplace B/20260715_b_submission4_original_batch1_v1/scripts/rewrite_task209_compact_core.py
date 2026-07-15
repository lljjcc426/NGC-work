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
    / "b_task209_dynamic_sprite_v3"
    / "task209.onnx"
)
DEFAULT_OUT = (
    ROOT
    / "reconstruction_candidates"
    / "b_task209_compact_core_v4"
    / "task209.onnx"
)


def rewrite(source: onnx.ModelProto) -> onnx.ModelProto:
    model = onnx.ModelProto.FromString(source.SerializeToString())
    nodes: list[onnx.NodeProto] = []
    changed: set[str] = set()
    for source_node in model.graph.node:
        output = source_node.output[0] if source_node.output else ""
        if output == "Bw":
            changed.add(output)
            continue
        if output == "ga_geom_rows":
            nodes.extend(
                [
                    helper.make_node(
                        "Gather",
                        ["strip", "ga_geom_idx"],
                        ["ga_geom_row_strip"],
                        axis=2,
                    ),
                    helper.make_node(
                        "Gather",
                        ["ga_geom_row_strip", "bwcols"],
                        ["ga_geom_rows"],
                        axis=3,
                    ),
                ]
            )
            changed.add(output)
            continue
        if output == "ga_row_ref":
            nodes.append(
                helper.make_node(
                    "Gather", ["ga_geom_rows", "ga_ref_idx"], [output], axis=2
                )
            )
            changed.add(output)
            continue
        if output == "ga_geom_cols":
            nodes.extend(
                [
                    helper.make_node(
                        "Gather",
                        ["bwcols", "ga_geom_idx"],
                        ["ga_geom_col_indices"],
                    ),
                    helper.make_node(
                        "Gather",
                        ["strip", "ga_geom_col_indices"],
                        ["ga_geom_cols"],
                        axis=3,
                    ),
                ]
            )
            changed.add(output)
            continue
        if output == "ga_col_ref":
            nodes.append(
                helper.make_node(
                    "Gather", ["ga_geom_cols", "ga_ref_idx"], [output], axis=3
                )
            )
            changed.add(output)
            continue
        if output in {"ga_sample_f16", "ga_legend_f16"}:
            nodes.append(
                helper.make_node(
                    "Cast", [source_node.input[0]], [output], to=TensorProto.UINT8
                )
            )
            changed.add(output)
            continue
        if output == "ga_align_score":
            nodes.append(
                helper.make_node(
                    "ConvInteger",
                    ["ga_legend_f16", "ga_sample_f16"],
                    [output],
                    pads=[0, 0, 2, 4],
                )
            )
            changed.add(output)
            continue
        if output == "ga_valid_f16":
            changed.add(output)
            continue
        if output == "ga_valid_score":
            nodes.append(
                helper.make_node(
                    "Where", ["ga_valid", "ga_align_score", "z0i"], [output]
                )
            )
            changed.add(output)
            continue
        nodes.append(onnx.NodeProto.FromString(source_node.SerializeToString()))

    required = {
        "Bw",
        "ga_geom_rows",
        "ga_row_ref",
        "ga_geom_cols",
        "ga_col_ref",
        "ga_sample_f16",
        "ga_legend_f16",
        "ga_align_score",
        "ga_valid_f16",
        "ga_valid_score",
    }
    if changed != required:
        raise RuntimeError(f"task209 compact-core mismatch: {sorted(required - changed)}")

    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.value_info[:]
    oe.prune_dead(model)
    oe.prune_initializers(model)
    for name, shape in (
        ("ds_legend_crop", [3, 5]),
        ("ds_sprite", [12, 12]),
        ("placed", [16, 20]),
        ("Oc", [16, 20]),
        ("Oidx30", [30, 30]),
    ):
        model.graph.value_info.append(
            helper.make_tensor_value_info(name, TensorProto.UINT8, shape)
        )
    onnx.checker.check_model(model, full_check=True)
    return model


def compare(base: Path, candidate: Path) -> dict[str, int]:
    base_session = ort.InferenceSession(
        base.read_bytes(), providers=["CPUExecutionProvider"]
    )
    candidate_session = ort.InferenceSession(
        candidate.read_bytes(), providers=["CPUExecutionProvider"]
    )
    examples = json.loads(
        (ROOT / "data" / "competition" / "task209.json").read_text()
    )
    checked = 0
    for split in ("train", "test", "arc-gen"):
        for example in examples.get(split, []):
            pair = build_blend.convert_to_numpy(example)
            if pair is None:
                continue
            expected = base_session.run(["output"], {"input": pair["input"]})[0]
            actual = candidate_session.run(["output"], {"input": pair["input"]})[0]
            if not np.array_equal(expected > 0, actual > 0):
                raise RuntimeError(
                    f"task209 compact-core equivalence failed in {split} example {checked}"
                )
            if not np.array_equal(actual > 0, pair["output"] > 0):
                raise RuntimeError(
                    f"task209 compact-core truth failed in {split} example {checked}"
                )
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
        "task": 209,
        "method": "direct strip geometry and uint8 integer alignment correlation",
        "equivalence": compare(args.base, args.out),
        "score": build_blend.validate_and_score(
            (209, "task209_compact_core", str(args.out))
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
