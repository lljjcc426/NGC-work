from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort


TASK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TASK_ROOT.parents[2]
COMMON = REPO_ROOT / "workplace C" / "neurogolf-2026-work" / "scripts" / "c_score_common.py"
OLD = TASK_ROOT / "onnx" / "task349_width29_toptrim_base.onnx"
NEW = TASK_ROOT / "onnx" / "task349_candidate.onnx"


def load_common():
    spec = importlib.util.spec_from_file_location("task349_c_score_common", COMMON)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    common = load_common()
    old = common.score_onnx("task349", OLD, validate_all=True)
    new = common.score_onnx("task349", NEW, validate_all=True)
    reports = TASK_ROOT / "reports"
    row = {
        "task": "task349",
        "old_cost": old.cost,
        "new_cost": new.cost,
        "delta_cost": new.cost - old.cost,
        "old_points": old.points,
        "new_points": new.points,
        "delta_points": new.points - old.points,
        "old_memory": old.memory,
        "new_memory": new.memory,
        "old_params": old.params,
        "new_params": new.params,
        "examples_checked": new.examples_checked,
        "examples_passed": new.examples_passed,
        "local_valid": new.ok,
        "accepted": new.ok and new.cost < old.cost,
        "artifact_path": str(NEW),
        "sha256": new.sha256,
    }
    with (reports / "cost_diff.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    utils = common.load_official_utils()
    payload = json.loads((common.TASK_DATA_DIR / "task349.json").read_text(encoding="utf-8"))
    session = ort.InferenceSession(NEW.read_bytes(), providers=["CPUExecutionProvider"])
    rows = []
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(payload[split]):
            arrays = utils.convert_to_numpy(example)
            mismatch = int(np.count_nonzero(utils.run_network(session, arrays["input"]) != arrays["output"]))
            rows.append({"split": split, "index": index, "passed": mismatch == 0, "mismatch_count": mismatch})
    with (reports / "rule_validation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "index", "passed", "mismatch_count"])
        writer.writeheader()
        writer.writerows(rows)
    print(row)


if __name__ == "__main__":
    main()
