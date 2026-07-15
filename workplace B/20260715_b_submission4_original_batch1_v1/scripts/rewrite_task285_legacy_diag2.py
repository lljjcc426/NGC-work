from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))
import build_blend  # noqa: E402
import optimize_equivalent as oe  # noqa: E402
import rewrite_task285_direct_orientation_or as previous  # noqa: E402


DEFAULT_BASE = previous.DEFAULT_BASE
DEFAULT_OUT = (
    ROOT
    / "reconstruction_candidates"
    / "b_task285_legacy_diag2_v7"
    / "task285.onnx"
)


def _node(op_type: str, inputs: list[str], output: str, **attrs) -> onnx.NodeProto:
    return helper.make_node(op_type, inputs, [output], name=output, **attrs)


def rewrite(source: onnx.ModelProto) -> onnx.ModelProto:
    model = previous.rewrite(source)
    diagonal = numpy_helper.from_array(
        np.array([[-31], [31]], dtype=np.int32),
        "diagonal_offsets",
    )
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name == "diagonal_offsets":
            model.graph.initializer[index].CopyFrom(diagonal)
            break
    else:
        raise RuntimeError("task285 diagonal_offsets initializer was not found")

    replacement = [
        _node("Cast", ["exact_same_bool"], "task285_same4_i8", to=TensorProto.INT8),
        _node("ReduceMax", ["task285_same4_i8", "s0"], "task285_has_same4", keepdims=0),
        _node(
            "Cast",
            ["exact_diagonal_same_bool"],
            "task285_same_diag_i8",
            to=TensorProto.INT8,
        ),
        _node(
            "ReduceMax",
            ["task285_same_diag_i8", "s0"],
            "task285_has_same_diag",
            keepdims=0,
        ),
        _node("Max", ["task285_has_same4", "task285_has_same_diag"], "exact_has_same"),
    ]
    nodes = list(model.graph.node)
    start = next(i for i, node in enumerate(nodes) if "exact_any_same_bool" in node.output)
    end = next(i for i, node in enumerate(nodes) if "exact_has_same" in node.output) + 1
    nodes[start:end] = replacement
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.value_info[:]
    oe.prune_dead(model)
    oe.prune_initializers(model)
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stress", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=285_20260714)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(rewrite(onnx.load(args.base)), args.out)
    checks = previous.previous.previous.previous.previous
    result = {
        "task": 285,
        "method": "orthogonal anchors with two-direction legacy compatibility",
        "equivalence": checks.official_equivalence(args.base, args.out),
        "stress": checks.stress_test(args.out, args.stress, args.seed) if args.stress else None,
        "score": build_blend.validate_and_score((285, "task285_legacy_diag2", str(args.out))),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
