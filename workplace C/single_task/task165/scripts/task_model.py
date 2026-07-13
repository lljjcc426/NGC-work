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


TASK = "task165"
HERE = Path(__file__).resolve()
TASK_DIR = HERE.parents[1]
REPO = HERE.parents[4]
COMMON = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"
DEFAULT_PARENT = Path(
    r"E:/kagglegolf/submissions/candidates/"
    r"GOLF_20260713_ALL399_DIRECT_13/onnx/task165.onnx"
)


def _set_cast_type(node: onnx.NodeProto, dtype: int) -> None:
    for attr in node.attribute:
        if attr.name == "to":
            attr.i = dtype
            return
    node.attribute.append(helper.make_attribute("to", dtype))


def build_integer_path(source: Path, output: Path, level: int) -> Path:
    model = deepcopy(onnx.load(str(source)))
    nodes = list(model.graph.node)
    if nodes[11].op_type != "Cast" or nodes[12].op_type != "Conv":
        raise RuntimeError("unexpected task165 parent graph")

    _set_cast_type(nodes[11], TensorProto.UINT8)
    old_conv = nodes[12]
    qconv = helper.make_node(
        "QLinearConv",
        [
            old_conv.input[0],
            "qsc",
            "qz",
            "Kshape_u8",
            "qsc",
            "qz",
            "qsc",
            "qz",
        ],
        list(old_conv.output),
        name=old_conv.name,
        pads=[0, 3, 0, 3],
    )
    nodes[12] = qconv
    if level == 1:
        qconv.output[0] = "shape_placed_u8"
        nodes.insert(
            13,
            helper.make_node(
                "Cast",
                ["shape_placed_u8"],
                ["shape_placed"],
                name="restore_shape_float16",
                to=TensorProto.FLOAT16,
            ),
        )
        greater_index = 14
        row_cast_index = 15
        where_index = 17
        final_cast_index = 18
    else:
        greater_index = 13
        row_cast_index = 14
        where_index = 16
        final_cast_index = 17
    greater = nodes[greater_index]
    greater.op_type = "Cast"
    del greater.input[:]
    greater.input.append("shape_placed")
    del greater.attribute[:]
    greater.attribute.append(helper.make_attribute("to", TensorProto.BOOL))

    if level >= 2:
        _set_cast_type(nodes[row_cast_index], TensorProto.UINT8)
        nodes[where_index].input[2] = "c99_u8"

    if level >= 3:
        nodes[where_index].output[0] = "botKite_u8"
        del nodes[final_cast_index]

    del model.graph.node[:]
    model.graph.node.extend(nodes)
    remove = {"Kshape", "half"}
    if level >= 2:
        remove.add("c99_f")
    kept = [item for item in model.graph.initializer if item.name not in remove]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.initializer.append(
        numpy_helper.from_array(
            np.array([[[[3, 2, 2, 1, 2, 2, 3]]]], dtype=np.uint8),
            name="Kshape_u8",
        )
    )
    del model.graph.value_info[:]
    model.producer_name = f"ngc_task165_integer_path_l{level}"
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
        "qconv_shape": build_integer_path(
            args.parent, debug / "task165_qconv_shape.onnx", 1
        ),
        "integer_row_path": build_integer_path(
            args.parent, debug / "task165_integer_row_path.onnx", 2
        ),
        "direct_u8_output": build_integer_path(
            args.parent, debug / "task165_direct_u8_output.onnx", 3
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
        if (
            result.ok
            and result.cost is not None
            and parent.cost is not None
            and result.cost < parent.cost
            and (best is None or result.cost < best[0])
        ):
            best = (result.cost, candidate)
    if best is not None:
        accepted = TASK_DIR / "onnx" / "task165_candidate.onnx"
        accepted.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best[1], accepted)
        print(json.dumps({"accepted": str(accepted), "cost": best[0]}))


if __name__ == "__main__":
    main()
