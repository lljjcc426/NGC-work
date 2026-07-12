from __future__ import annotations

import csv
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import onnx


TASK = "task237"
HERE = Path(__file__).resolve()
TASK_DIR = HERE.parents[1]
REPO = HERE.parents[4]
COMMON = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"
BASE = Path(r"E:/kagglegolf/submissions/downloaded_best/v93_7273_37_user_upload/onnx/task237.onnx")
DEBUG = TASK_DIR / "debug" / "task237_compact_packed_conv.onnx"
FINAL = TASK_DIR / "onnx" / "task237_candidate.onnx"


def build(output_path: Path = DEBUG) -> Path:
    model = deepcopy(onnx.load(BASE))
    nodes = list(model.graph.node)
    if len(nodes) != 36 or nodes[2].op_type != "Conv" or nodes[3].op_type != "Slice":
        raise RuntimeError("unexpected task237 baseline graph")
    packed = nodes[2]
    packed.output[0] = "packed_grid"
    packed.attribute.append(onnx.helper.make_attribute("pads", [0, 0, -21, 0]))
    del model.graph.node[:]
    model.graph.node.extend([*nodes[:3], *nodes[4:]])
    kept_vi = [item for item in model.graph.value_info if item.name != "packed_grid30"]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_vi)
    model.producer_name = "ngc_c_task237_compact_packed_conv"
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
        "method": "compact_packed_conv",
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
