from __future__ import annotations

import argparse
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


TASK = "task218"
HERE = Path(__file__).resolve()
TASK_DIR = HERE.parents[1]
REPO = HERE.parents[4]
COMMON = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"
DEFAULT_PARENT = Path(r"E:/kagglegolf/submissions/candidates/GOLF_20260713_ALL399_DIRECT_13/onnx/task218.onnx")


def build(source: Path, output: Path) -> Path:
    model = deepcopy(onnx.load(str(source)))
    tensor = next(x for x in model.graph.initializer if x.name == "channel_weights")
    weights = numpy_helper.to_array(tensor)[:, :, :1, :1].copy()
    kept = [x for x in model.graph.initializer if x.name != "channel_weights"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.initializer.append(numpy_helper.from_array(weights.astype(np.float32), "channel_weights"))
    node = next(n for n in model.graph.node if n.output[0] == "sample_grid")
    del node.attribute[:]
    node.attribute.extend([helper.make_attribute("pads", [0, 0, -9, -9]), helper.make_attribute("strides", [3, 3])])
    del model.graph.value_info[:]
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, str(output))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT); args = parser.parse_args()
    sys.path.insert(0, str(COMMON)); from c_score_common import score_onnx
    candidate = build(args.parent, TASK_DIR / "debug" / "task218_support_crop.onnx")
    parent = score_onnx(TASK, args.parent, validate_all=True); result = score_onnx(TASK, candidate, validate_all=True)
    print(json.dumps({"task": TASK, "valid": result.ok, "passed": result.examples_passed, "checked": result.examples_checked, "parent_cost": parent.cost, "candidate_cost": result.cost, "delta_cost": None if result.cost is None else parent.cost-result.cost, "sha256": result.sha256, "error": result.error}))
    if result.ok and result.cost is not None and result.cost < parent.cost:
        accepted = TASK_DIR / "onnx" / "task218_candidate.onnx"; accepted.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(candidate, accepted); print(json.dumps({"accepted": str(accepted), "cost": result.cost}))


if __name__ == "__main__": main()
