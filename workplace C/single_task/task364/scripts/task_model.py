from __future__ import annotations

import argparse
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import onnx
from onnx import helper


TASK = "task364"
HERE = Path(__file__).resolve()
TASK_DIR = HERE.parents[1]
REPO = HERE.parents[4]
COMMON = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"
DEFAULT_PARENT = Path(
    r"E:/kagglegolf/submissions/candidates/"
    r"GOLF_20260713_ALL399_DIRECT_13/onnx/task364.onnx"
)


def _source_parts(source: Path) -> tuple[onnx.ModelProto, list[onnx.NodeProto]]:
    model = deepcopy(onnx.load(str(source)))
    nodes = list(model.graph.node)
    expected = ["MaxPool", "Mul"] * 5
    if [node.op_type for node in nodes[14:24]] != expected:
        raise RuntimeError("unexpected task364 propagation graph")
    if nodes[14].input[0] != "e0" or nodes[23].output[0] != "SE5":
        raise RuntimeError("unexpected task364 propagation tensors")
    return model, nodes


def build_deferred_mask(source: Path, output: Path) -> Path:
    model, nodes = _source_parts(source)
    propagation: list[onnx.NodeProto] = []
    value = "e0"
    for round_index in range(1, 6):
        pooled = f"PE{round_index}"
        propagation.append(
            helper.make_node(
                "MaxPool",
                [value],
                [pooled],
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
            )
        )
        value = pooled
    propagation.append(helper.make_node("Mul", [value, "Gu"], ["SE5"]))
    del model.graph.node[:]
    model.graph.node.extend([*nodes[:14], *propagation, *nodes[24:]])
    model.producer_name = "ngc_task364_deferred_mask"
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(output))
    return output


def build_large_pool(source: Path, output: Path, kernels: tuple[int, ...]) -> Path:
    model, nodes = _source_parts(source)
    propagation: list[onnx.NodeProto] = []
    value = "e0"
    for index, kernel in enumerate(kernels, start=1):
        pooled = f"task364_pool_{index}"
        pad = kernel // 2
        propagation.append(
            helper.make_node(
                "MaxPool",
                [value],
                [pooled],
                kernel_shape=[kernel, kernel],
                pads=[pad, pad, pad, pad],
            )
        )
        value = pooled
    propagation.append(helper.make_node("Mul", [value, "Gu"], ["SE5"]))
    del model.graph.node[:]
    model.graph.node.extend([*nodes[:14], *propagation, *nodes[24:]])
    model.producer_name = "ngc_task364_large_pool"
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(output))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    args = parser.parse_args()

    sys.path.insert(0, str(COMMON))
    from c_score_common import score_onnx

    debug = TASK_DIR / "debug" / "propagation_rebuild"
    candidates = {
        "deferred_mask": build_deferred_mask(
            args.parent, debug / "task364_deferred_mask.onnx"
        ),
        "single_11x11": build_large_pool(
            args.parent, debug / "task364_single_11x11.onnx", (11,)
        ),
        "5x5_then_7x7": build_large_pool(
            args.parent, debug / "task364_5x5_then_7x7.onnx", (5, 7)
        ),
    }
    parent = score_onnx(TASK, args.parent, validate_all=True)
    best = None
    for name, path in candidates.items():
        result = score_onnx(TASK, path, validate_all=True)
        row = {
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
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if (
            result.ok
            and result.cost is not None
            and parent.cost is not None
            and result.cost < parent.cost
            and (best is None or result.cost < best[0])
        ):
            best = (result.cost, path)

    if best is not None:
        destination = TASK_DIR / "onnx" / "task364_candidate.onnx"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best[1], destination)
        print(json.dumps({"accepted": str(destination), "cost": best[0]}))


if __name__ == "__main__":
    main()
