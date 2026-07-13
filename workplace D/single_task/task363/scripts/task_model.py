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


TASK = "task363"
HERE = Path(__file__).resolve()
TASK_DIR = HERE.parents[1]
REPO = HERE.parents[4]
COMMON = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"
DEFAULT_PARENT = Path(
    r"E:/kagglegolf/submissions/candidates/"
    r"GOLF_20260713_ALL399_DIRECT_13/onnx/task363.onnx"
)


def _set_cast(node: onnx.NodeProto, dtype: int) -> None:
    for attr in node.attribute:
        if attr.name == "to":
            attr.i = dtype
            return
    raise RuntimeError(f"Cast node {node.output[0]} has no to attribute")


def _qconv(inputs: list[str], output: str, pads: list[int]) -> onnx.NodeProto:
    return helper.make_node(
        "QLinearConv",
        [
            inputs[0],
            "task363_qscale",
            "task363_qzero",
            inputs[1],
            "task363_qscale",
            "task363_qzero",
            "task363_qscale",
            "task363_qzero",
        ],
        [output],
        kernel_shape=[4, 4],
        pads=pads,
    )


def build_quantized(
    source: Path, output: Path, quantize_detection: bool, quantize_paint: bool
) -> Path:
    model = deepcopy(onnx.load(str(source)))
    rebuilt = []
    for node in model.graph.node:
        out = node.output[0]
        if quantize_detection and out == "black16":
            _set_cast(node, TensorProto.UINT8)
            rebuilt.append(node)
        elif quantize_detection and quantize_paint and out == "ker16":
            continue
        elif quantize_detection and out == "seed_count":
            rebuilt.append(
                _qconv(
                    ["ker_u8", "task363_seed_weight"],
                    "seed_count_u8",
                    [0, 0, 0, 0],
                )
            )
        elif quantize_detection and out == "score":
            rebuilt.append(_qconv(["black16", "ker_u8"], "score", [0, 0, 3, 3]))
        elif quantize_detection and out == "valid_b":
            node.input[1] = "seed_count_u8"
            rebuilt.append(node)
        elif quantize_paint and out == "valid16":
            _set_cast(node, TensorProto.UINT8)
            rebuilt.append(node)
        elif quantize_paint and out == "kerflip":
            node.input[0] = "ker_u8"
            rebuilt.append(node)
        elif quantize_paint and out == "paint_f":
            rebuilt.append(_qconv(["valid16", "kerflip"], "paint_f", [3, 3, 0, 0]))
        elif quantize_paint and out == "paint_b":
            source_name = node.input[0]
            rebuilt.append(helper.make_node("Cast", [source_name], ["paint_b"], to=TensorProto.BOOL))
        else:
            rebuilt.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rebuilt)
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(np.asarray(1.0, dtype=np.float32), "task363_qscale"),
            numpy_helper.from_array(np.asarray(0, dtype=np.uint8), "task363_qzero"),
        ]
    )
    if quantize_detection:
        model.graph.initializer.append(
            numpy_helper.from_array(
                np.ones((1, 1, 4, 4), dtype=np.uint8),
                "task363_seed_weight",
            )
        )
    used_initializers = {name for node in model.graph.node for name in node.input}
    kept_initializers = [
        item for item in model.graph.initializer if item.name in used_initializers
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)

    changed = set()
    if quantize_detection:
        changed.update({"black16", "score"})
    if quantize_paint:
        changed.update({"valid16", "kerflip", "paint_f"})
    found = set()
    for value_info in model.graph.value_info:
        if value_info.name in changed:
            value_info.type.tensor_type.elem_type = TensorProto.UINT8
            found.add(value_info.name)
    fixed_shapes = {
        "black16": [1, 1, 10, 10],
        "score": [1, 1, 10, 10],
        "valid16": [1, 1, 10, 10],
        "kerflip": [1, 1, 4, 4],
        "paint_f": [1, 1, 10, 10],
    }
    for name in changed - found:
        model.graph.value_info.append(
            helper.make_tensor_value_info(name, TensorProto.UINT8, fixed_shapes[name])
        )
    if quantize_detection:
        model.graph.value_info.append(
            helper.make_tensor_value_info("seed_count_u8", TensorProto.UINT8, [1, 1, 1, 1])
        )
    model.producer_name = "ngc_task363_quantized_dynamic_conv"
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

    debug = TASK_DIR / "debug"
    candidates = {
        "detection_qconv": build_quantized(
            args.parent, debug / "task363_detection_qconv.onnx", True, False
        ),
        "paint_qconv": build_quantized(
            args.parent, debug / "task363_paint_qconv.onnx", False, True
        ),
        "both_qconv": build_quantized(
            args.parent, debug / "task363_both_qconv.onnx", True, True
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
            "path": str(candidate),
            "error": result.error,
        }
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if result.ok and result.cost < parent.cost and (best is None or result.cost < best[0]):
            best = (result.cost, candidate)
    if best is not None:
        accepted = TASK_DIR / "onnx" / "task363_candidate.onnx"
        accepted.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best[1], accepted)
        print(json.dumps({"accepted": str(accepted), "cost": best[0]}))


if __name__ == "__main__":
    main()
