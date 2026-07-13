from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


TASK = "task315"
HERE = Path(__file__).resolve()
TASK_DIR = HERE.parents[1]
REPO = HERE.parents[4]
COMMON = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"
BASE = Path(
    r"E:/kagglegolf/submissions/candidates/"
    r"GOLF_20260712_102_submission5_plus_c22/onnx/task315.onnx"
)
DEBUG = TASK_DIR / "debug" / "task315_rank2_color_sign.onnx"
FINAL = TASK_DIR / "onnx" / "task315_candidate.onnx"


COLOR = np.array(
    [
        [4.0, 3.0],
        [-2.0, 4.0],
        [3.0, -2.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ],
    dtype=np.float32,
)

COEFF = np.array(
    [
        [[4.0, 2.0], [-1.0, 4.0]],
        [[1.0, 3.0], [1.0, 4.0]],
    ],
    dtype=np.float32,
)


def _replace(model: onnx.ModelProto, name: str, value: np.ndarray) -> None:
    for index, item in enumerate(model.graph.initializer):
        if item.name == name:
            model.graph.initializer[index].CopyFrom(
                numpy_helper.from_array(value, name=name)
            )
            return
    raise RuntimeError(f"missing initializer: {name}")


def build(output: Path = DEBUG) -> Path:
    model = onnx.load(str(BASE))
    if len(model.graph.node) != 1 or model.graph.node[0].op_type != "Einsum":
        raise RuntimeError("unexpected task315 baseline graph")
    _replace(model, "channel", COLOR)
    _replace(model, "coeff", COEFF)
    del model.graph.value_info[:]
    model.producer_name = "ngc_c_task315_rank2_color_sign"
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(output))
    return output


def main() -> None:
    sys.path.insert(0, str(COMMON))
    from c_score_common import score_onnx

    candidate = build()
    old = score_onnx(TASK, BASE, True)
    new = score_onnx(TASK, candidate, True)
    accepted = bool(
        new.ok
        and new.cost is not None
        and old.cost is not None
        and new.cost < old.cost
    )
    row = {
        "task": TASK,
        "method": "rank2_color_sign",
        "old_cost": old.cost,
        "new_cost": new.cost,
        "delta_cost": None if old.cost is None or new.cost is None else new.cost - old.cost,
        "old_points": old.points,
        "new_points": new.points,
        "delta_points": None if old.points is None or new.points is None else new.points - old.points,
        "examples_passed": new.examples_passed,
        "examples_checked": new.examples_checked,
        "local_valid": str(new.ok).lower(),
        "accepted": str(accepted).lower(),
        "artifact_path": str(candidate),
        "error": new.error,
    }
    report = TASK_DIR / "reports" / "cost_diff_round2.csv"
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    if accepted:
        shutil.copy2(candidate, FINAL)
    print(row)


if __name__ == "__main__":
    main()
