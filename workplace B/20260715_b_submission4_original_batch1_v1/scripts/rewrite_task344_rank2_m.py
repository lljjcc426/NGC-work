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


BASE = ROOT / "team_baselines" / "team_submission4_20260715" / "extracted" / "task344.onnx"
OUT = ROOT / "reconstruction_candidates" / "b_task344_rank2_m_v1" / "task344.onnx"


def rewrite(base: onnx.ModelProto) -> onnx.ModelProto:
    model = onnx.ModelProto.FromString(base.SerializeToString())
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    source = arrays["M"]
    u, singular, vt = np.linalg.svd(source.reshape(-1, 3), full_matrices=False)
    root = np.sqrt(singular[:2])
    left = (u[:, :2] * root).reshape(10, 10, 2).astype(np.float32)
    right = (root[:, None] * vt[:2]).astype(np.float32)

    kept = [item for item in model.graph.initializer if item.name != "M"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(left, "M_left"),
            numpy_helper.from_array(right, "M_right"),
        ]
    )

    if len(model.graph.node) != 1 or model.graph.node[0].op_type != "Einsum":
        raise RuntimeError("unexpected task344 graph")
    node = model.graph.node[0]
    expected = ["input", "E", "E", "M", "M", "E", "E", "U", "G", "A"]
    if list(node.input) != expected:
        raise RuntimeError(f"unexpected Einsum inputs: {list(node.input)}")
    del node.input[:]
    node.input.extend(
        [
            "input",
            "E",
            "E",
            "M_left",
            "M_right",
            "M_left",
            "M_right",
            "E",
            "E",
            "U",
            "G",
            "A",
        ]
    )
    equation = next(attr for attr in node.attribute if attr.name == "equation")
    equation.s = b"bnij,iv,jw,pvt,tl,qwu,ul,rp,sq,ok,nk,kl->bors"

    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    model = rewrite(onnx.load(BASE))
    onnx.save(model, OUT)
    result = build_blend.validate_and_score((344, "rank2_m", str(OUT)))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
