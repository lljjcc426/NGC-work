from __future__ import annotations

import csv
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


TASK = "task009"
HERE = Path(__file__).resolve()
TASK_DIR = HERE.parents[1]
REPO = HERE.parents[4]
COMMON = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"
BASE = TASK_DIR / "onnx" / "task009_outside_sentinel.onnx"
DEBUG = TASK_DIR / "debug" / "task009_compact_separator_pads.onnx"
FINAL = TASK_DIR / "onnx" / "task009_candidate.onnx"


def build(output_path: Path = DEBUG) -> Path:
    model = deepcopy(onnx.load(BASE))
    nodes = list(model.graph.node)
    pads = [node for node in nodes if node.op_type == "Pad" and node.output[0] in {"v_sep", "h_sep"}]
    if len(pads) != 2:
        raise RuntimeError("unexpected task009 separator Pad graph")
    for item in model.graph.initializer:
        if item.name in {"padc", "padr"}:
            item.CopyFrom(numpy_helper.from_array(np.array([0, 1], dtype=np.int64), name=item.name))
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(np.array([3], dtype=np.int64), name="compact_width_axis"),
            numpy_helper.from_array(np.array([2], dtype=np.int64), name="compact_height_axis"),
        ]
    )
    for node in pads:
        node.input.append("compact_width_axis" if node.output[0] == "v_sep" else "compact_height_axis")
    model.opset_import[0].version = 18
    model.producer_name = "ngc_c_task009_compact_separator_pads"
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
        "method": "compact_separator_pads",
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
