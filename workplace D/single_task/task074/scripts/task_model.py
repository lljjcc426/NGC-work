from __future__ import annotations

import argparse
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


TASK = "task074"
HERE = Path(__file__).resolve()
TASK_DIR = HERE.parents[1]
REPO = HERE.parents[4]
COMMON = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"
DEFAULT_PARENT = Path(
    r"E:/kagglegolf/submissions/candidates/GOLF_20260713_ALL399_DIRECT_13/onnx/task074.onnx"
)


def build(source: Path, output: Path) -> Path:
    model = deepcopy(onnx.load(str(source)))
    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    orbit_ids = initializers["cell_reps"].reshape(1, 1, 30, 30, 1).astype(np.int64)

    rebuilt = []
    for node in model.graph.node:
        if node.op_type == "Reshape" and node.output[0] == "flat_color":
            continue
        if node.op_type == "ScatterElements" and node.output[0] == "orbit_colors":
            rebuilt.append(
                helper.make_node(
                    "ScatterND",
                    ["base", "cell_reps", "color_u8"],
                    ["orbit_colors"],
                    reduction="max",
                )
            )
            continue
        if node.op_type == "Gather" and node.output[0] == "grid_flat":
            rebuilt.append(
                helper.make_node(
                    "Gather",
                    ["orbit_colors", "cell_reps_grid"],
                    ["grid_flat"],
                    axis=0,
                )
            )
            continue
        rebuilt.append(node)

    del model.graph.node[:]
    model.graph.node.extend(rebuilt)

    kept = [
        item
        for item in model.graph.initializer
        if item.name not in {"cell_reps", "base", "sh900"}
    ]
    kept.extend(
        [
            numpy_helper.from_array(orbit_ids, "cell_reps"),
            numpy_helper.from_array(np.zeros((136,), dtype=np.uint8), "base"),
        ]
    )
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    del model.graph.value_info[:]

    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.save(model, str(output))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    args = parser.parse_args()
    sys.path.insert(0, str(COMMON))
    from c_score_common import score_onnx

    candidate = build(args.parent, TASK_DIR / "debug" / "task074_scatternd.onnx")
    parent_score = score_onnx(TASK, args.parent, validate_all=True)
    candidate_score = score_onnx(TASK, candidate, validate_all=True)
    result = {
        "task": TASK,
        "valid": candidate_score.ok,
        "passed": candidate_score.examples_passed,
        "checked": candidate_score.examples_checked,
        "parent_cost": parent_score.cost,
        "candidate_cost": candidate_score.cost,
        "delta_cost": (
            None
            if candidate_score.cost is None
            else parent_score.cost - candidate_score.cost
        ),
        "sha256": candidate_score.sha256,
        "error": candidate_score.error,
    }
    print(json.dumps(result))
    if (
        candidate_score.ok
        and candidate_score.cost is not None
        and candidate_score.cost < parent_score.cost
    ):
        accepted = TASK_DIR / "onnx" / "task074_candidate.onnx"
        accepted.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, accepted)
        print(json.dumps({"accepted": str(accepted), "cost": candidate_score.cost}))


if __name__ == "__main__":
    main()
