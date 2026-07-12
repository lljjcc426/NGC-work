from __future__ import annotations

import csv
import sys
from copy import deepcopy
from pathlib import Path

import onnx
from onnx import helper


TASK = "task072"
HERE = Path(__file__).resolve()
TASK_DIR = HERE.parents[1]
REPO = HERE.parents[4]
COMMON = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"
BASE = Path(r"E:/kagglegolf/submissions/downloaded_best/v93_7273_37_user_upload/onnx/task072.onnx")
OUT = TASK_DIR / "onnx" / "task072_legacy_attr_controls.onnx"


def build_onnx(path: Path = OUT) -> Path:
    model = deepcopy(onnx.load(BASE))
    old = list(model.graph.node)
    if [node.op_type for node in old] != ["Slice", "Slice", "Equal", "Where", "Pad"]:
        raise RuntimeError("unexpected task072 graph")
    nodes = [
        helper.make_node("Slice", ["input"], ["t"], starts=[2, 0, 0], ends=[3, 6, 5], axes=[1, 2, 3], name="top_attr_slice"),
        helper.make_node("Slice", ["input"], ["b"], starts=[2, 7, 0], ends=[3, 13, 5], axes=[1, 2, 3], name="bottom_attr_slice"),
        deepcopy(old[2]),
        deepcopy(old[3]),
        helper.make_node("Pad", ["p"], ["output"], pads=[0, 0, 0, 0, 0, 6, 24, 25], mode="constant", value=0.0, name="output_attr_pad"),
    ]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    keep = {"tv", "fv"}
    initializers = [item for item in model.graph.initializer if item.name in keep]
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    del model.opset_import[:]
    model.opset_import.extend([helper.make_opsetid("", 9)])
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
        "task": TASK, "method": "legacy_attr_controls", "old_cost": old.cost,
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
