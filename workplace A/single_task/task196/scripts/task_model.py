from __future__ import annotations

import argparse
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import onnx
from onnx import helper


TASK = "task196"
HERE = Path(__file__).resolve()
TASK_DIR = HERE.parents[1]
REPO = HERE.parents[4]
COMMON = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"
DEFAULT_PARENT = Path(
    r"E:/kagglegolf/submissions/candidates/GOLF_20260713_SUBMISSION6_REBASE/onnx/task196.onnx"
)


def build(source: Path, output: Path, kernels: tuple[int, ...]) -> Path:
    model = deepcopy(onnx.load(str(source)))
    prefix = list(model.graph.node[:3])
    suffix = list(model.graph.node[12:])
    current = "bad"
    propagation = []
    for index, kernel in enumerate(kernels):
        pool_output = "poolL" if index == len(kernels) - 1 else f"pool_new_{index}"
        radius = kernel // 2
        propagation.append(
            helper.make_node(
                "MaxPool",
                [current],
                [pool_output],
                kernel_shape=[kernel, kernel],
                pads=[radius, radius, radius, radius],
                strides=[1, 1],
            )
        )
        if index != len(kernels) - 1:
            masked = f"bad_new_{index}"
            propagation.append(helper.make_node("Min", [pool_output, "blue"], [masked]))
            current = masked

    del model.graph.node[:]
    model.graph.node.extend(prefix + propagation + suffix)
    del model.graph.value_info[:]
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.save(model, str(output))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    args = parser.parse_args()
    sys.path.insert(0, str(COMMON))
    from c_score_common import score_onnx

    parent_score = score_onnx(TASK, args.parent, validate_all=True)
    variants = {
        "four_local_rounds": (3, 3, 3, 3),
        "radii_1_2_2": (3, 5, 5),
        "radii_2_2_2": (5, 5, 5),
        "radii_1_1_3": (3, 3, 7),
        "radii_2_3": (5, 7),
    }
    best = None
    for name, kernels in variants.items():
        candidate = build(args.parent, TASK_DIR / "debug" / f"task196_{name}.onnx", kernels)
        score = score_onnx(TASK, candidate, validate_all=True)
        result = {
            "variant": name,
            "kernels": kernels,
            "valid": score.ok,
            "passed": score.examples_passed,
            "checked": score.examples_checked,
            "parent_cost": parent_score.cost,
            "candidate_cost": score.cost,
            "delta_cost": None if score.cost is None else parent_score.cost - score.cost,
            "sha256": score.sha256,
            "error": score.error,
        }
        print(json.dumps(result))
        if score.ok and score.cost is not None and score.cost < parent_score.cost:
            if best is None or score.cost < best[0]:
                best = (score.cost, candidate)

    if best is not None:
        accepted = TASK_DIR / "onnx" / "task196_candidate.onnx"
        accepted.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best[1], accepted)
        print(json.dumps({"accepted": str(accepted), "cost": best[0]}))


if __name__ == "__main__":
    main()
