from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper as oh, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import build_blend  # noqa: E402
import optimize_equivalent as oe  # noqa: E402


BASE = ROOT / "team_baselines" / "team_submission4_20260715" / "extracted" / "task270.onnx"
OUT = ROOT / "reconstruction_candidates" / "b_task270_prefix_directions_v4" / "task270.onnx"


def _axis_nodes(axis: str, petals: str, center: str) -> list[onnx.NodeProto]:
    before = "up2" if axis == "row" else "lf2"
    after = "dn2" if axis == "row" else "rt2"
    return [
        oh.make_node("CumSum", [petals, "direction_axis_scalar"], [f"{axis}_prefix"]),
        oh.make_node(
            "Einsum",
            [f"{axis}_prefix", center, "direction_one"],
            [f"{axis}_prefix_at_center"],
            equation="fr,fr,z->fz",
        ),
        oh.make_node(
            "Einsum",
            [petals, center, "direction_one"],
            [f"{axis}_petals_at_center"],
            equation="fr,fr,z->fz",
        ),
        oh.make_node(
            "ReduceSum",
            [petals, "ax1"],
            [f"{axis}_petals_total"],
            keepdims=1,
        ),
        oh.make_node(
            "Mul",
            [f"{axis}_petals_total", "direction_center_scale"],
            [f"{axis}_petals_total_scaled"],
        ),
        oh.make_node(
            "Sub",
            [f"{axis}_prefix_at_center", f"{axis}_petals_at_center"],
            [before],
        ),
        oh.make_node(
            "Sub",
            [f"{axis}_petals_total_scaled", f"{axis}_prefix_at_center"],
            [after],
        ),
    ]


def rewrite(base: onnx.ModelProto) -> onnx.ModelProto:
    model = onnx.ModelProto.FromString(base.SerializeToString())
    remove_outputs = {"up", "dn", "lf", "rt", "up2", "dn2", "lf2", "rt2"}
    kept: list[onnx.NodeProto] = []
    insertion = None
    removed: set[str] = set()
    for node in model.graph.node:
        if len(node.output) == 1 and node.output[0] in remove_outputs:
            removed.add(node.output[0])
            if insertion is None:
                insertion = len(kept)
            continue
        kept.append(node)
    if removed != remove_outputs or insertion is None:
        raise RuntimeError(f"unexpected removed direction outputs: {sorted(removed)}")

    compact = _axis_nodes("row", "rowP", "rc_f")
    compact.extend(_axis_nodes("col", "colP", "cc_f"))
    kept[insertion:insertion] = compact
    del model.graph.node[:]
    model.graph.node.extend(kept)
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(np.array(1, dtype=np.int64), "direction_axis_scalar"),
            numpy_helper.from_array(np.array([1], dtype=np.float16), "direction_one"),
            numpy_helper.from_array(
                np.array([[2], [1]], dtype=np.float16), "direction_center_scale"
            ),
        ]
    )

    oe.prune_dead(model)
    oe.prune_initializers(model)
    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    model = rewrite(onnx.load(BASE))
    onnx.save(model, OUT)
    result = build_blend.validate_and_score((270, "prefix_directions", str(OUT)))
    result["nodes"] = len(model.graph.node)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
