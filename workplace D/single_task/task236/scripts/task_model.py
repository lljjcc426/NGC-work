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


TASK = "task236"
HERE = Path(__file__).resolve()
TASK_DIR = HERE.parents[1]
REPO = HERE.parents[4]
COMMON = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"
DEFAULT_PARENT = Path(
    r"E:/kagglegolf/submissions/candidates/"
    r"GOLF_20260713_ALL399_DIRECT_13/onnx/task236.onnx"
)


def build_absorbed_shrink(source: Path, output: Path) -> Path:
    model = deepcopy(onnx.load(str(source)))
    weights = next(x for x in model.graph.initializer if x.name == "w_pair")
    array = numpy_helper.to_array(weights).copy()
    remap = {-128.0: -1.0, -1.0: 126.0, 0.0: 0.0, 127.0: 0.0}
    array = np.vectorize(remap.__getitem__, otypes=[np.float32])(array)
    replacement = numpy_helper.from_array(array.astype(np.float32), name="w_pair")
    kept = [x for x in model.graph.initializer if x.name != "w_pair"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.initializer.append(replacement)

    rebuilt = []
    for node in model.graph.node:
        if node.op_type == "Cast" and node.output[0] == "parity_code_i8":
            node.output[0] = "signed_parity_i8"
            rebuilt.append(node)
        elif node.op_type == "Shrink" and node.output[0] == "signed_parity_i8":
            continue
        else:
            rebuilt.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rebuilt)
    del model.graph.value_info[:]
    model.producer_name = "ngc_task236_absorbed_shrink"
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, str(output))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    args = parser.parse_args()
    sys.path.insert(0, str(COMMON))
    from c_score_common import score_onnx

    candidate = build_absorbed_shrink(
        args.parent, TASK_DIR / "debug" / "task236_absorbed_shrink.onnx"
    )
    parent = score_onnx(TASK, args.parent, validate_all=True)
    result = score_onnx(TASK, candidate, validate_all=True)
    record = {
        "task": TASK,
        "candidate": "absorbed_shrink",
        "valid": result.ok,
        "passed": result.examples_passed,
        "checked": result.examples_checked,
        "parent_cost": parent.cost,
        "candidate_cost": result.cost,
        "delta_cost": None if result.cost is None else parent.cost - result.cost,
        "sha256": result.sha256,
        "error": result.error,
    }
    print(json.dumps(record, ensure_ascii=False), flush=True)
    if result.ok and result.cost is not None and result.cost < parent.cost:
        accepted = TASK_DIR / "onnx" / "task236_candidate.onnx"
        accepted.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, accepted)
        print(json.dumps({"accepted": str(accepted), "cost": result.cost}))


if __name__ == "__main__":
    main()
