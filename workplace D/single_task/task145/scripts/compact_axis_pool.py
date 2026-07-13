from __future__ import annotations

import argparse
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import onnx


TASK = "task145"
HERE = Path(__file__).resolve()
TASK_DIR = HERE.parents[1]
REPO = HERE.parents[4]
COMMON = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"
DEFAULT_PARENT = Path(
    r"E:/kagglegolf/submissions/candidates/"
    r"GOLF_20260713_SUBMISSION8_ARCHIVE_LOCAL_REBASE/onnx/task145.onnx"
)


def build(source: Path, output: Path) -> Path:
    model = deepcopy(onnx.load(str(source)))
    row_pool = next(node for node in model.graph.node if node.output[0] == "row_inside_f")
    col_pool = next(node for node in model.graph.node if node.output[0] == "col_inside_f")
    row_cast = next(node for node in model.graph.node if node.output[0] == "row_inside")
    col_cast = next(node for node in model.graph.node if node.output[0] == "col_inside")

    compact = onnx.helper.make_node(
        "Cast", ["z_bool"], ["z_u8"], name="compact_z_u8", to=onnx.TensorProto.UINT8
    )
    insert_at = min(
        index
        for index, node in enumerate(model.graph.node)
        if node is row_pool or node is col_pool
    )
    model.graph.node.insert(insert_at, compact)
    row_pool.input[0] = "z_u8"
    col_pool.input[0] = "z_u8"
    row_pool.output[0] = "row_inside_u8"
    col_pool.output[0] = "col_inside_u8"
    row_cast.input[0] = "row_inside_u8"
    col_cast.input[0] = "col_inside_u8"

    del model.graph.value_info[:]
    model.producer_name = "ngc_task145_compact_axis_pool"
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

    probe = build(args.parent, TASK_DIR / "debug" / "task145_compact_axis_pool.onnx")
    parent = score_onnx(TASK, args.parent, validate_all=True)
    candidate = score_onnx(TASK, probe, validate_all=True)
    accepted = bool(
        candidate.ok
        and candidate.cost is not None
        and parent.cost is not None
        and candidate.cost < parent.cost
    )
    if accepted:
        destination = TASK_DIR / "onnx" / "task145_candidate.onnx"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(probe, destination)
    print(
        json.dumps(
            {
                "task": TASK,
                "method": "uint8_axis_pool",
                "parent_cost": parent.cost,
                "candidate_cost": candidate.cost,
                "delta_cost": None
                if parent.cost is None or candidate.cost is None
                else parent.cost - candidate.cost,
                "examples_passed": candidate.examples_passed,
                "examples_checked": candidate.examples_checked,
                "accepted": accepted,
                "sha256": candidate.sha256,
                "error": candidate.error,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
