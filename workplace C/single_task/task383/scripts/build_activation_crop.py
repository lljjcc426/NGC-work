from __future__ import annotations

import csv
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


TASK = "task383"
HERE = Path(__file__).resolve()
TASK_DIR = HERE.parents[1]
REPO = HERE.parents[4]
COMMON = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"
BASE = TASK_DIR / "onnx" / "task383_conv_crop_collapse.onnx"
DEBUG = TASK_DIR / "debug" / "task383_activation_crop.onnx"
FINAL = TASK_DIR / "onnx" / "task383_candidate.onnx"


def _replace(model: onnx.ModelProto, name: str, value: np.ndarray) -> None:
    for index, item in enumerate(model.graph.initializer):
        if item.name == name:
            model.graph.initializer[index].CopyFrom(numpy_helper.from_array(value, name=name))
            return
    model.graph.initializer.append(numpy_helper.from_array(value, name=name))


def build(height: int, width: int, output_path: Path = DEBUG) -> Path:
    model = deepcopy(onnx.load(BASE))
    nodes = list(model.graph.node)
    if len(nodes) != 54 or nodes[0].op_type != "Conv" or nodes[52].op_type != "Pad":
        raise RuntimeError("unexpected task383 compact baseline graph")
    for attribute in nodes[0].attribute:
        if attribute.name == "pads":
            attribute.ints[:] = [0, 0, height - 30, width - 30]

    _replace(model, "rev_row", np.arange(height - 1, -1, -1, dtype=np.int64))
    _replace(model, "rev_col", np.arange(width - 1, -1, -1, dtype=np.int64))
    _replace(model, "last_row", np.array([height - 1], dtype=np.int32))
    _replace(model, "last_col", np.array([width - 1], dtype=np.int32))
    _replace(model, "sr", np.array([1, 1, height, 1], dtype=np.int64))
    _replace(model, "sc", np.array([1, 1, 1, width], dtype=np.int64))
    _replace(model, "pad", np.array([0, 0, 0, 0, 0, 0, 30 - height, 30 - width], dtype=np.int64))
    nodes[12].input[1] = "rev_row"
    nodes[13].input[1] = "rev_col"
    nodes[20].input[0] = "last_row"
    nodes[21].input[0] = "last_col"

    used = {name for node in nodes for name in node.input}
    kept = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    del model.graph.value_info[:]
    model.producer_name = "ngc_c_task383_activation_crop"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(output_path))
    return output_path


def main() -> None:
    sys.path.insert(0, str(COMMON))
    from c_score_common import score_onnx

    old = score_onnx(TASK, BASE, True)
    rows = []
    accepted = []
    for height, width in ((23, 24), (24, 23), (23, 22)):
        candidate = DEBUG.with_name(f"task383_activation_crop_{height}x{width}.onnx")
        build(height, width, candidate)
        new = score_onnx(TASK, candidate, True)
        row = {
            "task": TASK,
            "method": f"activation_crop_{height}x{width}",
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
        rows.append(row)
        if new.ok and new.cost is not None and old.cost is not None and new.cost < old.cost:
            accepted.append((new.cost, candidate))
        print(row)
    report = TASK_DIR / "reports" / "cost_diff_round2.csv"
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if accepted:
        _, candidate = min(accepted)
        FINAL.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, FINAL)
        print(FINAL)


if __name__ == "__main__":
    main()
