from __future__ import annotations

import csv
import sys
from copy import deepcopy
from pathlib import Path

import onnx


TASK = "task347"
HERE = Path(__file__).resolve()
TASK_DIR = HERE.parents[1]
REPO = HERE.parents[4]
COMMON = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"
BASE = Path(r"E:/kagglegolf/submissions/downloaded_best/v93_7273_37_user_upload/onnx/task347.onnx")
OUT = TASK_DIR / "onnx" / "task347_inferred_unpool_shape.onnx"


def build_onnx(path: Path = OUT) -> Path:
    model = deepcopy(onnx.load(BASE))
    unpool = model.graph.node[-1]
    if unpool.op_type != "MaxUnpool" or list(unpool.input) != ["small2", "unpool_indices", "output_shape"]:
        raise RuntimeError("unexpected task347 MaxUnpool structure")
    del unpool.input[:]
    unpool.input.extend(["small2", "unpool_indices"])
    del unpool.attribute[:]
    unpool.attribute.extend([
        onnx.helper.make_attribute("kernel_shape", [1, 1]),
        onnx.helper.make_attribute("strides", [15, 15]),
        onnx.helper.make_attribute("pads", [0, 0, 1, 1]),
    ])
    kept = [item for item in model.graph.initializer if item.name != "output_shape"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, path)
    return path


def main() -> None:
    sys.path.insert(0, str(COMMON))
    from c_score_common import score_onnx

    candidate = build_onnx()
    old = score_onnx(TASK, BASE, True)
    new = score_onnx(TASK, candidate, True)
    row = {
        "task": TASK, "method": "inferred_unpool_shape", "old_cost": old.cost,
        "new_cost": new.cost, "delta_cost": None if old.cost is None or new.cost is None else new.cost - old.cost,
        "examples_passed": new.examples_passed, "examples_checked": new.examples_checked,
        "local_valid": str(new.ok).lower(), "accepted": str(bool(new.ok and new.cost < old.cost)).lower(),
        "artifact_path": str(candidate),
    }
    report = TASK_DIR / "reports" / "cost_diff_round2.csv"
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
    print(row)


if __name__ == "__main__":
    main()
