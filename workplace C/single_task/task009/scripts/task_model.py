from __future__ import annotations

import argparse
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


TASK = "task009"
HERE = Path(__file__).resolve()
TASK_DIR = HERE.parents[1]
REPO = HERE.parents[4]
COMMON = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"
DEFAULT_PARENT = Path(
    r"E:/kagglegolf/submissions/candidates/"
    r"GOLF_20260713_ALL399_DIRECT_13/onnx/task009.onnx"
)


def build_uint8_einsum(source: Path, output: Path) -> Path:
    model = deepcopy(onnx.load(str(source)))
    rebuilt = []
    for node in model.graph.node:
        if node.op_type == "Cast" and node.output[0] == "onehot16":
            next(attr for attr in node.attribute if attr.name == "to").i = TensorProto.UINT8
            rebuilt.append(node)
        elif node.op_type == "Einsum" and node.output[0] == "h_between":
            node.output[0] = "h_u8"
            rebuilt.append(node)
        elif node.op_type == "Einsum" and node.output[0] == "v_between":
            node.output[0] = "v_u8"
            rebuilt.append(node)
        elif node.op_type == "Cast" and node.output[0] in {"h_u8", "v_u8"}:
            continue
        else:
            rebuilt.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rebuilt)

    replacements = {
        "cv16": np.arange(1, 10, dtype=np.uint8),
        "sepA": np.triu(np.ones((10, 10), dtype=np.uint8), 1),
    }
    kept = [x for x in model.graph.initializer if x.name not in replacements]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    for name, array in replacements.items():
        model.graph.initializer.append(numpy_helper.from_array(array, name=name))
    del model.graph.value_info[:]
    model.producer_name = "ngc_task009_uint8_einsum"
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

    candidate = build_uint8_einsum(
        args.parent, TASK_DIR / "debug" / "task009_uint8_einsum.onnx"
    )
    parent = score_onnx(TASK, args.parent, validate_all=True)
    result = score_onnx(TASK, candidate, validate_all=True)
    record = {
        "task": TASK,
        "candidate": "uint8_einsum",
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
        accepted = TASK_DIR / "onnx" / "task009_candidate.onnx"
        accepted.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, accepted)
        print(json.dumps({"accepted": str(accepted), "cost": result.cost}))


if __name__ == "__main__":
    main()
