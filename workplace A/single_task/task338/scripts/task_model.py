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


TASK = "task338"
HERE = Path(__file__).resolve()
TASK_DIR = HERE.parents[1]
REPO = HERE.parents[4]
COMMON = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"
DEFAULT_PARENT = Path(
    r"E:/kagglegolf/submissions/candidates/"
    r"GOLF_20260713_ALL399_DIRECT_13/onnx/task338.onnx"
)


def build_qconv_prefix(source: Path, output: Path) -> Path:
    model = deepcopy(onnx.load(str(source)))
    rebuilt = []
    for node in model.graph.node:
        if node.op_type == "Cast" and node.output[0] == "VE_f16":
            next(attr for attr in node.attribute if attr.name == "to").i = TensorProto.UINT8
            node.output[0] = "VE_u8"
            rebuilt.append(node)
        elif node.op_type == "CumSum" and node.output[0] == "cross_f16":
            rebuilt.append(
                helper.make_node(
                    "QLinearConv",
                    [
                        "VE_u8",
                        "task338_qscale",
                        "zero_u8",
                        "task338_prefix_kernel",
                        "task338_qscale",
                        "zero_u8",
                        "task338_qscale",
                        "zero_u8",
                    ],
                    ["cross_u8"],
                    name="task338_uint8_prefix",
                    pads=[0, 23, 0, 0],
                )
            )
        elif node.op_type == "Cast" and node.output[0] == "cross_u8":
            continue
        else:
            rebuilt.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rebuilt)
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(np.asarray(1.0, dtype=np.float32), "task338_qscale"),
            numpy_helper.from_array(
                np.ones((1, 1, 1, 24), dtype=np.uint8),
                "task338_prefix_kernel",
            ),
        ]
    )
    used = {name for node in model.graph.node for name in node.input}
    kept = [x for x in model.graph.initializer if x.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    del model.graph.value_info[:]
    model.producer_name = "ngc_task338_qconv_prefix"
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, str(output))
    return output


def crop_detection_support(source: Path, output: Path) -> Path:
    model = deepcopy(onnx.load(str(source)))
    tensor = next(x for x in model.graph.initializer if x.name == "conv_w")
    weights = numpy_helper.to_array(tensor)
    cropped = weights[:, :, :3, :1].copy()
    kept = [x for x in model.graph.initializer if x.name != "conv_w"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.initializer.append(
        numpy_helper.from_array(cropped.astype(np.float32), "conv_w")
    )
    node = next(n for n in model.graph.node if n.output[0] == "ve_sum")
    del node.attribute[:]
    node.attribute.extend(
        [
            helper.make_attribute("kernel_shape", [3, 1]),
            helper.make_attribute("pads", [0, 0, -5, -6]),
        ]
    )
    del model.graph.value_info[:]
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

    support = crop_detection_support(
        args.parent, TASK_DIR / "debug" / "task338_support_crop.onnx"
    )
    candidates = {
        "qconv_prefix": build_qconv_prefix(
            args.parent, TASK_DIR / "debug" / "task338_qconv_prefix.onnx"
        ),
        "support_crop_qconv_prefix": build_qconv_prefix(
            support, TASK_DIR / "debug" / "task338_support_crop_qconv_prefix.onnx"
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
        accepted = TASK_DIR / "onnx" / "task338_candidate.onnx"
        accepted.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best[1], accepted)
        print(json.dumps({"accepted": str(accepted), "cost": best[0]}))


if __name__ == "__main__":
    main()
