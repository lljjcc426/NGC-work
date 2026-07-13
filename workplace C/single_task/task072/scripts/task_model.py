from __future__ import annotations

import argparse
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


TASK = "task072"
HERE = Path(__file__).resolve()
TASK_DIR = HERE.parents[1]
REPO = HERE.parents[4]
COMMON = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"
DEFAULT_PARENT = Path(
    r"E:/kagglegolf/submissions/candidates/"
    r"GOLF_20260713_ALL399_DIRECT_13/onnx/task072.onnx"
)


def build_terminal_grouped_conv(source: Path, output: Path) -> Path:
    parent = onnx.load(str(source))
    diff = numpy_helper.to_array(
        next(x for x in parent.graph.initializer if x.name == "diff_w")
    ).copy()
    diff[diff == 200.0] = -56.0

    weights = np.zeros((10, 5, 2, 1), dtype=np.float32)
    weights[0] = -diff[0, :5]
    weights[3] = diff[0, :5]
    node = helper.make_node(
        "Conv",
        ["input", "task072_fused_w"],
        ["output"],
        name="task072_terminal_grouped_conv",
        dilations=[7, 1],
        group=2,
        kernel_shape=[2, 1],
        pads=[0, 0, 7, 0],
    )
    graph = helper.make_graph(
        [node],
        "task072_terminal_grouped_conv",
        list(parent.graph.input),
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        [numpy_helper.from_array(weights, name="task072_fused_w")],
    )
    model = helper.make_model(
        graph,
        producer_name="ngc_task072_terminal_grouped_conv",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    model.ir_version = min(parent.ir_version, 10)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, str(output))
    return output


def build_float_chain(source: Path, output: Path) -> Path:
    model = deepcopy(onnx.load(str(source)))
    diff_tensor = next(x for x in model.graph.initializer if x.name == "diff_w")
    diff = numpy_helper.to_array(diff_tensor).copy()
    diff[diff == 200.0] = -56.0
    paint_tensor = next(x for x in model.graph.initializer if x.name == "paint_w")
    paint = numpy_helper.to_array(paint_tensor).astype(np.float32)
    kept = [x for x in model.graph.initializer if x.name not in {"diff_w", "paint_w"}]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(diff.astype(np.float32), name="diff_w"),
            numpy_helper.from_array(paint, name="paint_w_f32"),
        ]
    )
    rebuilt = []
    for node in model.graph.node:
        if node.op_type == "Cast" and node.output[0] == "delta_i8":
            continue
        if node.op_type == "ConvInteger":
            node.op_type = "Conv"
            node.input[0] = "delta"
            node.input[1] = "paint_w_f32"
        rebuilt.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rebuilt)
    model.graph.output[0].type.tensor_type.elem_type = TensorProto.FLOAT
    del model.graph.value_info[:]
    model.producer_name = "ngc_task072_float_chain"
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

    candidates = {
        "terminal_grouped_conv": build_terminal_grouped_conv(
            args.parent, TASK_DIR / "debug" / "task072_terminal_grouped_conv.onnx"
        ),
        "float_chain": build_float_chain(
            args.parent, TASK_DIR / "debug" / "task072_float_chain.onnx"
        ),
    }
    parent = score_onnx(TASK, args.parent, validate_all=True)
    best = None
    for name, candidate in candidates.items():
        result = score_onnx(TASK, candidate, validate_all=True)
        record = {
            "task": TASK,
            "candidate": name,
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
            if best is None or result.cost < best[0]:
                best = (result.cost, candidate)
    if best is not None:
        accepted = TASK_DIR / "onnx" / "task072_candidate.onnx"
        accepted.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best[1], accepted)
        print(json.dumps({"accepted": str(accepted), "cost": best[0]}))


if __name__ == "__main__":
    main()
