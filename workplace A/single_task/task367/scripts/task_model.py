from __future__ import annotations

import argparse
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import onnx
from onnx import helper


TASK = "task367"
HERE = Path(__file__).resolve()
TASK_DIR = HERE.parents[1]
REPO = HERE.parents[4]
COMMON = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"
DEFAULT_PARENT = Path(
    r"E:/kagglegolf/submissions/candidates/"
    r"GOLF_20260713_SUBMISSION8_REBASE/onnx/task367.onnx"
)


def _load(source: Path) -> onnx.ModelProto:
    model = deepcopy(onnx.load(str(source)))
    outputs = [node.output[0] for node in model.graph.node]
    expected = {"v_a1", "v_a2", "v_eq", "v_zb", "v_fill"}
    if not expected.issubset(outputs):
        raise RuntimeError("unexpected task367 parent graph")
    return model


def _save(model: onnx.ModelProto, output: Path, producer: str) -> Path:
    # Shape inference must describe only tensors still produced by this graph.
    del model.graph.value_info[:]
    model.producer_name = producer
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, str(output))
    return output


def build_fused_sum(source: Path, output: Path) -> Path:
    model = _load(source)
    rebuilt = []
    for node in model.graph.node:
        if node.output[0] == "v_a1":
            continue
        if node.output[0] == "v_a2":
            rebuilt.append(
                helper.make_node("Max", ["v_col", "v_s1h", "v_s1v"], ["v_a2"])
            )
        else:
            rebuilt.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rebuilt)
    return _save(model, output, "ngc_task367_fused_sum")


def build_direct_fill(source: Path, output: Path) -> Path:
    model = _load(source)
    rebuilt = []
    for node in model.graph.node:
        if node.output[0] in {"v_eq", "v_zb"}:
            continue
        if node.output[0] == "v_fill":
            rebuilt.append(helper.make_node("Less", ["r5c", "v_z"], ["v_fill"]))
        else:
            rebuilt.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rebuilt)
    return _save(model, output, "ngc_task367_direct_fill")


def build_combined(source: Path, output: Path) -> Path:
    model = _load(source)
    rebuilt = []
    for node in model.graph.node:
        if node.output[0] in {"v_a1", "v_eq", "v_zb"}:
            continue
        if node.output[0] == "v_a2":
            rebuilt.append(
                helper.make_node("Max", ["v_col", "v_s1h", "v_s1v"], ["v_a2"])
            )
        elif node.output[0] == "v_fill":
            rebuilt.append(helper.make_node("Less", ["r5c", "v_z"], ["v_fill"]))
        else:
            rebuilt.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rebuilt)
    return _save(model, output, "ngc_task367_fused_sum_direct_fill")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    args = parser.parse_args()

    sys.path.insert(0, str(COMMON))
    from c_score_common import score_onnx

    debug = TASK_DIR / "debug"
    candidates = {
        "fused_sum": build_fused_sum(
            args.parent, debug / "task367_fused_sum.onnx"
        ),
        "direct_fill": build_direct_fill(
            args.parent, debug / "task367_direct_fill.onnx"
        ),
        "combined": build_combined(
            args.parent, debug / "task367_combined.onnx"
        ),
    }
    parent = score_onnx(TASK, args.parent, validate_all=True)
    best = None
    for name, path in candidates.items():
        result = score_onnx(TASK, path, validate_all=True)
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
            "path": str(path),
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
            best = (result.cost, path)

    if best is not None:
        accepted = TASK_DIR / "onnx" / "task367_candidate.onnx"
        accepted.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best[1], accepted)
        print(json.dumps({"accepted": str(accepted), "cost": best[0]}))


if __name__ == "__main__":
    main()
