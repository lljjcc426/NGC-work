from __future__ import annotations

from pathlib import Path

import onnx
from onnx import helper


TASK_ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(
    r"E:/kagglegolf/submissions/candidates/"
    r"GOLF_20260709_101_prvsiyan_7266_72_repro/onnx/task054.onnx"
)


def build(output: Path) -> Path:
    """Draw marker-safe lines first, then restore template neighborhoods."""
    model = onnx.load(str(SOURCE))
    removed = {"safe_name_311", "safe_name_318", "safe_name_319", "safe_name_337", "safe_name_338"}
    nodes = [node for node in model.graph.node if not any(name in removed for name in node.output)]
    final = nodes.pop()
    if final.op_type != "Equal" or list(final.output) != ["output"]:
        raise RuntimeError("unexpected task054 graph ending")

    fused = [
        helper.make_node("Reshape", ["safe_name_167", "safe_name_16"], ["task054_line_color"]),
        helper.make_node("Gather", ["safe_name_320", "safe_name_308"], ["task054_h_existing_4d"], axis=2),
        helper.make_node("Reshape", ["task054_h_existing_4d", "safe_name_24"], ["task054_h_existing"]),
        helper.make_node("Gather", ["safe_name_92", "safe_name_308"], ["task054_h_base_4d"], axis=2),
        helper.make_node("Reshape", ["task054_h_base_4d", "safe_name_24"], ["task054_h_base"]),
        helper.make_node("Greater", ["safe_name_307", "task054_h_base"], ["task054_h_condition"]),
        helper.make_node(
            "Where", ["task054_h_condition", "task054_line_color", "task054_h_existing"], ["task054_h_updates"]
        ),
        helper.make_node(
            "ScatterND", ["safe_name_320", "safe_name_310", "task054_h_updates"], ["task054_line_h"]
        ),
        helper.make_node(
            "GatherElements", ["task054_line_h", "safe_name_317"], ["task054_v_existing"], axis=3
        ),
        helper.make_node("GatherElements", ["safe_name_92", "safe_name_317"], ["task054_v_base"], axis=3),
        helper.make_node("Greater", ["safe_name_313", "task054_v_base"], ["task054_v_condition"]),
        helper.make_node(
            "Where", ["task054_v_condition", "task054_line_color", "task054_v_existing"], ["task054_v_updates"]
        ),
        helper.make_node(
            "ScatterElements",
            ["task054_line_h", "safe_name_317", "task054_v_updates"],
            ["task054_line_v"],
            axis=3,
            reduction="max",
        ),
        helper.make_node(
            "ScatterND", ["task054_line_v", "safe_name_329", "safe_name_336"], ["safe_name_338"]
        ),
    ]
    del model.graph.node[:]
    model.graph.node.extend(nodes + fused + [final])

    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(output))
    return output


if __name__ == "__main__":
    print(build(TASK_ROOT / "onnx" / "task054_line_then_neighbor_candidate.onnx"))
