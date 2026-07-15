from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import build_blend  # noqa: E402
import optimize_equivalent as oe  # noqa: E402


BASE = (
    ROOT
    / "public_probe_variants"
    / "team_submission5_b_work_20260716"
    / "submission"
    / "task295.onnx"
)
OUT = (
    ROOT
    / "reconstruction_candidates"
    / "b_task295_precontract_reuse_v2"
    / "task295.onnx"
)


def build() -> onnx.ModelProto:
    model = onnx.load(BASE)
    initializers = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    kernel = np.einsum(
        "tu,tv,tqn->uvqn",
        initializers["VA"],
        initializers["GA"],
        initializers["Hc"],
        optimize=True,
    ).astype(np.float32)
    rounded = np.rint(kernel)
    kernel = np.where(np.abs(kernel - rounded) < 1e-6, rounded, kernel).astype(np.float32)
    selector = kernel.sum(axis=(1, 2, 3))
    if not np.array_equal(selector, np.array([-1.0, 0.0], dtype=np.float32)):
        raise RuntimeError(f"unexpected contracted selector: {selector}")
    model.graph.initializer.append(numpy_helper.from_array(kernel, "K_uvqn"))

    gb = initializers["GB"].copy()
    gb[:, 1, :] *= -1.0
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name == "GB":
            model.graph.initializer[index].CopyFrom(numpy_helper.from_array(gb, "GB"))
            break

    found_lt = False
    found_output = False
    for node in model.graph.node:
        if list(node.output) == ["Lt"]:
            if node.op_type != "Einsum":
                raise RuntimeError(f"unexpected Lt op: {node.op_type}")
            del node.input[:]
            node.input.extend(["input", "VB", "K_uvqn"])
            equation = next(attr for attr in node.attribute if attr.name == "equation")
            equation.s = b"bchw,uc,uvqn->b"
            found_lt = True
        elif list(node.output) == ["output"]:
            if node.op_type != "Einsum":
                raise RuntimeError(f"unexpected output op: {node.op_type}")
            del node.input[:]
            node.input.extend(
                ["input", "VB", "sc", "K_uvqn", "GB", "B3", "g", "sw", "B3"]
            )
            equation = next(attr for attr in node.attribute if attr.name == "equation")
            equation.s = b"bshw,us,p,uvqn,vpm,mr,r,q,nc->bsrc"
            found_output = True
    if not found_lt or not found_output:
        raise RuntimeError(f"rewrite incomplete: Lt={found_lt}, output={found_output}")

    oe.prune_initializers(model)
    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    model = build()
    onnx.save(model, OUT)
    result = build_blend.validate_and_score((295, "precontract_reuse", str(OUT)))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
