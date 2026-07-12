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
COMMON_PATH = REPO_ROOT / "workplace C" / "neurogolf-2026-work" / "scripts" / "c_score_common.py"
BASELINE = Path(
    r"E:/kagglegolf/submissions/candidates/"
    r"GOLF_20260709_101_prvsiyan_7266_72_repro/onnx/task193.onnx"
)
CANDIDATE = TASK_ROOT / "onnx" / "task193_candidate.onnx"


def load_common():
    spec = importlib.util.spec_from_file_location("task193_c_score_common", COMMON_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    common = load_common()
    old = common.score_onnx("task193", BASELINE, validate_all=True)
    new = common.score_onnx("task193", CANDIDATE, validate_all=True)
    reports = TASK_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    fields = [
        "task", "old_cost", "new_cost", "delta_cost", "old_points",
        "new_points", "delta_points", "examples_checked", "examples_passed",
        "local_valid", "accepted", "artifact_path", "sha256",
    ]
    row = {
        "task": "task193",
        "old_cost": old.cost,
        "new_cost": new.cost,
        "delta_cost": new.cost - old.cost,
        "old_points": old.points,
        "new_points": new.points,
        "delta_points": new.points - old.points,
        "examples_checked": new.examples_checked,
        "examples_passed": new.examples_passed,
        "local_valid": new.ok,
        "accepted": new.ok and new.cost < old.cost,
        "artifact_path": str(CANDIDATE),
        "sha256": new.sha256,
    }
    with (reports / "cost_diff.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)

    utils = common.load_official_utils()
    examples = json.loads((common.TASK_DATA_DIR / "task193.json").read_text(encoding="utf-8"))
    session = ort.InferenceSession(CANDIDATE.read_bytes(), providers=["CPUExecutionProvider"])
    validation_rows = []
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(examples.get(split, [])):
            arrays = utils.convert_to_numpy(example)
            predicted = utils.run_network(session, arrays["input"])
            mismatch = int(np.count_nonzero(predicted != arrays["output"]))
            validation_rows.append(
                {"split": split, "index": index, "passed": mismatch == 0, "mismatch_count": mismatch}
            )
    with (reports / "rule_validation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "index", "passed", "mismatch_count"])
        writer.writeheader()
        writer.writerows(validation_rows)
    print(row)


if __name__ == "__main__":
    main()
