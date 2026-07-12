from __future__ import annotations

import csv
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


TASK = "task091"
HERE = Path(__file__).resolve()
TASK_DIR = HERE.parents[1]
REPO = HERE.parents[4]
COMMON = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"
BASE = Path(r"E:/kagglegolf/submissions/downloaded_best/v93_7273_37_user_upload/onnx/task091.onnx")
DEBUG = TASK_DIR / "debug" / "task091_compact_pad_axes.onnx"
FINAL = TASK_DIR / "onnx" / "task091_candidate.onnx"


def build(output_path: Path = DEBUG) -> Path:
    model = deepcopy(onnx.load(BASE))
    nodes = list(model.graph.node)
    if len(nodes) != 50 or nodes[47].op_type != "Concat" or nodes[48].op_type != "Pad":
        raise RuntimeError("unexpected task091 baseline graph")
    zero6 = next(item for item in model.graph.initializer if item.name == "zero6")
    zero6.CopyFrom(numpy_helper.from_array(np.zeros(2, dtype=np.int64), name="zero6"))
    model.graph.initializer.append(numpy_helper.from_array(np.array([2, 3], dtype=np.int64), name="compact_pad_axes"))
    nodes[48].input.append("compact_pad_axes")
    axes_names: dict[tuple[int, ...], str] = {}
    for node in nodes:
        if node.op_type not in {"Squeeze", "Unsqueeze"}:
            continue
        attribute = next((item for item in node.attribute if item.name == "axes"), None)
        if attribute is None:
            continue
        values = tuple(int(value) for value in attribute.ints)
        name = axes_names.setdefault(values, "compact_axes_" + "_".join(map(str, values)))
        if not any(item.name == name for item in model.graph.initializer):
            model.graph.initializer.append(numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name))
        kept = [item for item in node.attribute if item.name != "axes"]
        del node.attribute[:]
        node.attribute.extend(kept)
        node.input.append(name)
    for opset in model.opset_import:
        if opset.domain in {"", "ai.onnx"}:
            opset.version = max(opset.version, 18)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    kept_vi = [item for item in model.graph.value_info if item.name != "pads"]
    kept_vi.append(helper.make_tensor_value_info("pads", TensorProto.INT64, [4]))
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_vi)
    model.producer_name = "ngc_c_task091_compact_pad_axes"
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
        "method": "compact_pad_axes",
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
