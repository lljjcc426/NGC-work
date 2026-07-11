from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import onnx


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_ROOT = REPO_ROOT / "workplace C" / "neurogolf-2026-work" / "scripts"
BASE_ONNX = (
    Path("E:/kagglegolf")
    / "submissions"
    / "candidates"
    / "GOLF_20260709_101_prvsiyan_7266_72_repro"
    / "onnx"
    / "task356.onnx"
)
DEBUG_ONNX = REPO_ROOT / "workplace C" / "single_task" / "task356" / "debug" / "task356_zero_tie_probe.onnx"
FINAL_ONNX = REPO_ROOT / "workplace C" / "single_task" / "task356" / "onnx" / "task356_candidate.onnx"


def build(output_path: Path) -> Path:
    model = onnx.load(str(BASE_ONNX))
    nodes = list(model.graph.node)
    add_nodes = [node for node in nodes if node.op_type == "Add" and list(node.output) == ["code"]]
    if len(add_nodes) != 1:
        raise RuntimeError("expected exactly one Add producing code")
    conv_nodes = [node for node in nodes if node.op_type == "ConvInteger" and list(node.output) == ["output"]]
    if len(conv_nodes) != 1:
        raise RuntimeError("expected exactly one ConvInteger producing output")

    new_nodes = []
    for node in nodes:
        if node is add_nodes[0]:
            continue
        copied = onnx.NodeProto()
        copied.CopyFrom(node)
        if copied.op_type == "ConvInteger" and list(copied.output) == ["output"]:
            copied.input[:] = ["mask10", "conv_weights"]
        new_nodes.append(copied)

    kept_initializers = [
        init for init in model.graph.initializer if init.name != "conv_x_zero_point"
    ]
    graph = onnx.helper.make_graph(
        new_nodes,
        model.graph.name,
        model.graph.input,
        model.graph.output,
        kept_initializers,
        value_info=[vi for vi in model.graph.value_info if vi.name != "code"],
    )
    candidate = onnx.helper.make_model(graph, opset_imports=model.opset_import, ir_version=model.ir_version)
    candidate.producer_name = "ngc_c_task356_zero_tie_fold"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(candidate)
    onnx.save(candidate, str(output_path))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()

    built = build(DEBUG_ONNX)
    print(built)
    sys.path.insert(0, str(SCRIPT_ROOT))
    from c_score_common import score_onnx

    result = score_onnx("task356", built, validate_all=True)
    print(
        {
            "ok": result.ok,
            "examples": f"{result.examples_passed}/{result.examples_checked}",
            "memory": result.memory,
            "params": result.params,
            "cost": result.cost,
            "points": result.points,
            "error": result.error,
        }
    )
    if args.promote:
        if not result.ok or result.cost is None or result.cost >= 1319:
            raise SystemExit("refusing to promote non-improving task356 candidate")
        FINAL_ONNX.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built, FINAL_ONNX)
        print(FINAL_ONNX)


if __name__ == "__main__":
    main()
