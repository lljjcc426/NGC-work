from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

import build_blend


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "team_baselines" / "team_submission2_20260713" / "submission" / "task350.onnx"
DEFAULT_OUT = ROOT / "reconstruction_candidates" / "b_task350_shared_rules_v1" / "task350.onnx"


def rewrite(base: onnx.ModelProto) -> onnx.ModelProto:
    model = onnx.ModelProto.FromString(base.SerializeToString())
    if len(model.graph.node) != 1 or model.graph.node[0].op_type != "Einsum":
        raise RuntimeError("task350 baseline no longer has the expected single Einsum")

    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    r1 = initializers["R1"]
    r2 = initializers["R2"]
    r0_selector = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
    r2_selector = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    r2_basis = np.stack([r2[0], r2[2]])

    r0 = initializers["R0"]
    if not np.array_equal(r0, np.einsum("ht,tab->hab", r0_selector, r1)):
        raise RuntimeError("task350 R0 sharing identity no longer holds")
    if not np.array_equal(r2, np.einsum("ht,tab->hab", r2_selector, r2_basis)):
        raise RuntimeError("task350 R2 factorization no longer holds")

    kept = [item for item in model.graph.initializer if item.name not in {"R0", "R2"}]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(r0_selector, "R0_selector"),
            numpy_helper.from_array(r2_selector, "R2_selector"),
            numpy_helper.from_array(r2_basis, "R2_basis"),
        ]
    )

    node = model.graph.node[0]
    equation = next(attr.s.decode() for attr in node.attribute if attr.name == "equation")
    terms, output = equation.split("->")
    terms_list = terms.split(",")
    latent_r0 = iter("STUV")
    latent_r2 = iter("WXYZ")
    new_inputs: list[str] = []
    new_terms: list[str] = []
    for name, term in zip(node.input, terms_list, strict=True):
        if name == "R0":
            latent = next(latent_r0)
            new_inputs.extend(["R1", "R0_selector"])
            new_terms.extend([latent + term[1:], term[0] + latent])
        elif name == "R2":
            latent = next(latent_r2)
            new_inputs.extend(["R2_basis", "R2_selector"])
            new_terms.extend([latent + term[1:], term[0] + latent])
        else:
            new_inputs.append(name)
            new_terms.append(term)

    replacement = helper.make_node(
        "Einsum",
        new_inputs,
        ["output"],
        name="output",
        equation=",".join(new_terms) + "->" + output,
    )
    del model.graph.node[:]
    model.graph.node.append(replacement)
    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    return model


def compare_signs(base_path: Path, candidate_path: Path) -> dict[str, int]:
    base_session = ort.InferenceSession(base_path.read_bytes(), providers=["CPUExecutionProvider"])
    candidate_session = ort.InferenceSession(candidate_path.read_bytes(), providers=["CPUExecutionProvider"])
    examples = json.loads((ROOT / "data" / "competition" / "task350.json").read_text())
    checked = 0
    for split in ("train", "test", "arc-gen"):
        for example in examples.get(split, []):
            pair = build_blend.convert_to_numpy(example)
            if pair is None:
                continue
            expected = base_session.run(["output"], {"input": pair["input"]})[0] > 0
            actual = candidate_session.run(["output"], {"input": pair["input"]})[0] > 0
            if not np.array_equal(expected, actual):
                return {"right": checked, "total": checked + 1}
            checked += 1
    return {"right": checked, "total": checked}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(rewrite(onnx.load(args.base)), args.out)
    result = {
        "score": build_blend.validate_and_score((350, "task350_shared_rules", str(args.out))),
        "official_sign_equivalence": compare_signs(args.base, args.out),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
