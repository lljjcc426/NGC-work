from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


TASK_ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(
    r"E:/kagglegolf/submissions/candidates/"
    r"GOLF_20260709_101_prvsiyan_7266_72_repro/onnx/task054.onnx"
)


def build(output: Path) -> Path:
    """Deduplicate compact line records, then write scalar lines after overlays."""
    model = onnx.load(str(SOURCE))
    model.graph.initializer.append(numpy_helper.from_array(np.array(0, dtype=np.int32), name="task054_zero_i32"))
    removed = {"safe_name_311", "safe_name_318", "safe_name_319", "safe_name_338"}
    nodes = [node for node in model.graph.node if not any(name in removed for name in node.output)]
    final = nodes.pop()
    if final.op_type != "Equal" or list(final.output) != ["output"]:
        raise RuntimeError("unexpected task054 graph ending")

    fused = [
        helper.make_node("Reshape", ["safe_name_167", "safe_name_16"], ["task054_line_color"]),
        # Union the four horizontal masks for every duplicate row index.
        helper.make_node("Reshape", ["safe_name_308", "safe_name_23"], ["task054_rows_col"]),
        helper.make_node("Reshape", ["safe_name_308", "safe_name_20"], ["task054_rows_row"]),
        helper.make_node("Equal", ["task054_rows_col", "task054_rows_row"], ["task054_row_equal"]),
        helper.make_node("Cast", ["task054_row_equal"], ["task054_row_equal_u8"], to=TensorProto.UINT8),
        helper.make_node("MatMulInteger", ["task054_row_equal_u8", "safe_name_307"], ["task054_h_counts"]),
        helper.make_node("Greater", ["task054_h_counts", "task054_zero_i32"], ["task054_h_union"]),
        helper.make_node("Gather", ["safe_name_337", "safe_name_308"], ["task054_h_existing_4d"], axis=2),
        helper.make_node("Reshape", ["task054_h_existing_4d", "safe_name_24"], ["task054_h_existing"]),
        helper.make_node("Gather", ["safe_name_92", "safe_name_308"], ["task054_h_base_4d"], axis=2),
        helper.make_node("Reshape", ["task054_h_base_4d", "safe_name_24"], ["task054_h_base"]),
        helper.make_node("Equal", ["task054_h_base", "safe_name_2"], ["task054_h_background"]),
        helper.make_node("And", ["task054_h_union", "task054_h_background"], ["task054_h_condition"]),
        helper.make_node(
            "Where", ["task054_h_condition", "task054_line_color", "task054_h_existing"], ["task054_h_updates"]
        ),
        helper.make_node("ScatterND", ["safe_name_337", "safe_name_310", "task054_h_updates"], ["task054_line_h"]),
        # Apply the same collision-safe union to vertical masks and columns.
        helper.make_node("Reshape", ["safe_name_316", "safe_name_23"], ["task054_cols_col"]),
        helper.make_node("Reshape", ["safe_name_316", "safe_name_20"], ["task054_cols_row"]),
        helper.make_node("Equal", ["task054_cols_col", "task054_cols_row"], ["task054_col_equal"]),
        helper.make_node("Cast", ["task054_col_equal"], ["task054_col_equal_u8"], to=TensorProto.UINT8),
        helper.make_node("Reshape", ["safe_name_312", "safe_name_24"], ["task054_v_masks"]),
        helper.make_node("MatMulInteger", ["task054_col_equal_u8", "task054_v_masks"], ["task054_v_counts"]),
        helper.make_node("Greater", ["task054_v_counts", "task054_zero_i32"], ["task054_v_union_4x30"]),
        helper.make_node("Transpose", ["task054_v_union_4x30"], ["task054_v_union_30x4"], perm=[1, 0]),
        helper.make_node("Reshape", ["task054_v_union_30x4", "safe_name_19"], ["task054_v_union"]),
        helper.make_node("GatherElements", ["task054_line_h", "safe_name_317"], ["task054_v_existing"], axis=3),
        helper.make_node("GatherElements", ["safe_name_92", "safe_name_317"], ["task054_v_base"], axis=3),
        helper.make_node("Equal", ["task054_v_base", "safe_name_2"], ["task054_v_background"]),
        helper.make_node("And", ["task054_v_union", "task054_v_background"], ["task054_v_condition"]),
        helper.make_node(
            "Where", ["task054_v_condition", "task054_line_color", "task054_v_existing"], ["task054_v_updates"]
        ),
        helper.make_node(
            "ScatterElements",
            ["task054_line_h", "safe_name_317", "task054_v_updates"],
            ["safe_name_338"],
            axis=3,
        ),
    ]
    del model.graph.node[:]
    model.graph.node.extend(nodes + fused + [final])
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(output))
    return output


if __name__ == "__main__":
    print(build(TASK_ROOT / "onnx" / "task054_candidate.onnx"))
