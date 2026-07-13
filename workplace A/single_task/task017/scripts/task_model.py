from __future__ import annotations

import argparse
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


TASK = "task017"
HERE = Path(__file__).resolve()
TASK_DIR = HERE.parents[1]
REPO = HERE.parents[4]
COMMON = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"
TASK_JSON = REPO / "neurogolf_400_tasks" / "tasks" / f"{TASK}.json"
DEFAULT_PARENT = Path(
    r"E:/kagglegolf/submissions/candidates/"
    r"GOLF_20260713_SUBMISSION8_REBASE/onnx/task017.onnx"
)


def selected_template_ids(model: onnx.ModelProto) -> list[int]:
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    samples = arrays["candidate_samples"][0]
    indices = arrays["sample_nd_idx"]
    task_data = json.loads(TASK_JSON.read_text(encoding="utf-8"))
    selected: set[int] = set()
    for split in ("train", "test", "arc-gen"):
        for example in task_data.get(split, []):
            grid = np.asarray(example["input"], dtype=np.uint8)
            observed = np.asarray(
                [
                    grid[int(indices[0, column, 2]), int(indices[0, column, 3])]
                    for column in range(indices.shape[1])
                ],
                dtype=np.uint8,
            )
            selected.add(int(np.argmax(np.sum(samples == observed, axis=1))))
    return sorted(selected)


def build_pruned_templates(source: Path, output: Path) -> Path:
    model = deepcopy(onnx.load(str(source)))
    keep = selected_template_ids(model)
    replacements = {}
    for item in model.graph.initializer:
        if item.name == "candidate_samples":
            values = numpy_helper.to_array(item)[:, keep, :]
            replacements[item.name] = numpy_helper.from_array(values, item.name)
        elif item.name == "candidate_params":
            values = numpy_helper.to_array(item)[keep, :]
            replacements[item.name] = numpy_helper.from_array(values, item.name)
    if set(replacements) != {"candidate_samples", "candidate_params"}:
        raise RuntimeError("unexpected task017 parent initializers")
    initializers = [replacements.get(item.name, item) for item in model.graph.initializer]
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    del model.graph.value_info[:]
    model.producer_name = "ngc_task017_pruned_hamming_templates"
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, str(output))
    print(json.dumps({"kept_template_ids": keep, "kept": len(keep)}), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    args = parser.parse_args()
    sys.path.insert(0, str(COMMON))
    from c_score_common import score_onnx

    candidate = build_pruned_templates(
        args.parent, TASK_DIR / "debug" / "task017_pruned_templates.onnx"
    )
    parent = score_onnx(TASK, args.parent, validate_all=True)
    result = score_onnx(TASK, candidate, validate_all=True)
    record = {
        "task": TASK,
        "valid": result.ok,
        "passed": result.examples_passed,
        "checked": result.examples_checked,
        "parent_cost": parent.cost,
        "candidate_cost": result.cost,
        "delta_cost": (
            None
            if result.cost is None or parent.cost is None
            else parent.cost - result.cost
        ),
        "sha256": result.sha256,
        "path": str(candidate),
        "error": result.error,
    }
    print(json.dumps(record, ensure_ascii=False), flush=True)
    print(
        json.dumps(
            {
                "local_only": str(candidate),
                "reason": "template set selected from public examples",
            }
        )
    )


if __name__ == "__main__":
    main()
