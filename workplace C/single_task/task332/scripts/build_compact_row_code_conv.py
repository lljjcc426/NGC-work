from __future__ import annotations

import csv
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


TASK = "task332"
HERE = Path(__file__).resolve()
TASK_DIR = HERE.parents[1]
REPO = HERE.parents[4]
COMMON = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"
BASE = Path(r"E:/kagglegolf/submissions/downloaded_best/v93_7273_37_user_upload/onnx/task332.onnx")
DEBUG = TASK_DIR / "debug" / "task332_compact_row_code_conv.onnx"
FINAL = TASK_DIR / "onnx" / "task332_candidate.onnx"


def build(output_path: Path = DEBUG) -> Path:
    model = deepcopy(onnx.load(BASE))
    nodes = list(model.graph.node)
    if len(nodes) != 11 or nodes[0].op_type != "Einsum" or nodes[1].op_type != "Slice":
        raise RuntimeError("unexpected task332 baseline graph")

    weight = np.zeros((1, 10, 3, 1), dtype=np.float32)
    weight[0, 5, :, 0] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    model.graph.initializer.append(numpy_helper.from_array(weight, name="compact_row_code_weight"))
    compact = helper.make_node(
        "Conv",
        ["input", "compact_row_code_weight"],
        ["crop"],
        kernel_shape=[3, 1],
        pads=[0, 0, -27, -10],
    )
    del model.graph.node[:]
    model.graph.node.extend([compact, *nodes[2:]])

    used = {name for node in model.graph.node for name in node.input}
    kept = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    kept_vi = [item for item in model.graph.value_info if item.name != "cr30"]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_vi)
    model.producer_name = "ngc_c_task332_compact_row_code_conv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(output_path))
    return output_path


def main() -> None:
    sys.path.insert(0, str(COMMON))
    from c_score_common import score_onnx

    candidate = build()
    old = score_onnx(TASK, BASE, True)
    new = score_onnx(TASK, candidate, True)
    row = {
        "task": TASK,
        "method": "compact_row_code_conv",
        "old_cost": old.cost,
        "new_cost": new.cost,
        "delta_cost": None if old.cost is None or new.cost is None else new.cost - old.cost,
        "old_points": old.points,
        "new_points": new.points,
        "delta_points": None if old.points is None or new.points is None else new.points - old.points,
        "examples_passed": new.examples_passed,
        "examples_checked": new.examples_checked,
        "local_valid": str(new.ok).lower(),
        "accepted": str(bool(new.ok and new.cost is not None and old.cost is not None and new.cost < old.cost)).lower(),
        "artifact_path": str(candidate),
        "error": new.error,
    }
    report = TASK_DIR / "reports" / "cost_diff_round2.csv"
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(row)
    if new.ok and new.cost is not None and old.cost is not None and new.cost < old.cost:
        FINAL.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, FINAL)
        print(FINAL)


if __name__ == "__main__":
    main()
