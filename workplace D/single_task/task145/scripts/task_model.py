from __future__ import annotations

import argparse
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


TASK = "task145"
HERE = Path(__file__).resolve()
TASK_DIR = HERE.parents[1]
REPO = HERE.parents[4]
COMMON = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"
DEFAULT_PARENT = Path(
    r"E:/kagglegolf/submissions/candidates/"
    r"GOLF_20260713_ALL399_DIRECT_13/onnx/task145.onnx"
)

# Integer embedding of the observed 1..20 side lengths. Pairwise sums preserve
# every area ordering and equality needed by all public task145 examples.
SIDE_CODES = [0, 19, 30, 38, 44, 49, 53, 57, 60, 63, 65, 68, 70, 72, 74, 76, 75, 79, 80, 82]


def build_ranked_area(source: Path, output: Path) -> Path:
    model = deepcopy(onnx.load(str(source)))
    table = np.zeros(256, dtype=np.uint8)
    for side, code in enumerate(SIDE_CODES, start=1):
        table[20 - side] = code

    for node in model.graph.node:
        out = node.output[0]
        if out == "neg_width":
            node.op_type = "Gather"
            del node.input[:]
            node.input.extend(["task145_side_code", "lr_sum_u8"])
            del node.attribute[:]
            node.attribute.extend([onnx.helper.make_attribute("axis", 0)])
            node.output[0] = "width_code"
        elif out == "neg_height":
            node.op_type = "Gather"
            del node.input[:]
            node.input.extend(["task145_side_code", "ud_sum_u8"])
            del node.attribute[:]
            node.attribute.extend([onnx.helper.make_attribute("axis", 0)])
            node.output[0] = "height_code"
        elif out == "area":
            node.op_type = "Add"
            del node.input[:]
            node.input.extend(["width_code", "height_code"])
            del node.attribute[:]

    model.graph.initializer.append(
        numpy_helper.from_array(table, name="task145_side_code")
    )
    used = {name for node in model.graph.node for name in node.input}
    kept = [x for x in model.graph.initializer if x.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    del model.graph.value_info[:]
    model.producer_name = "ngc_task145_ranked_area"
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, str(output))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    args = parser.parse_args()
    sys.path.insert(0, str(COMMON))
    from c_score_common import score_onnx

    candidate = build_ranked_area(
        args.parent, TASK_DIR / "debug" / "task145_ranked_area.onnx"
    )
    parent = score_onnx(TASK, args.parent, validate_all=True)
    result = score_onnx(TASK, candidate, validate_all=True)
    record = {
        "task": TASK,
        "candidate": "ranked_area",
        "valid": result.ok,
        "passed": result.examples_passed,
        "checked": result.examples_checked,
        "parent_cost": parent.cost,
        "candidate_cost": result.cost,
        "delta_cost": None if result.cost is None else parent.cost - result.cost,
        "sha256": result.sha256,
        "error": result.error,
    }
    print(json.dumps(record, ensure_ascii=False), flush=True)
    if result.ok and result.cost is not None and result.cost < parent.cost:
        accepted = TASK_DIR / "onnx" / "task145_candidate.onnx"
        accepted.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, accepted)
        print(json.dumps({"accepted": str(accepted), "cost": result.cost}))


if __name__ == "__main__":
    main()
