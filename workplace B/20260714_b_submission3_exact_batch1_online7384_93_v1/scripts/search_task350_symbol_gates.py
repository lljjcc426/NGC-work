from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import build_blend  # noqa: E402


BASE = ROOT / "team_baselines" / "team_submission3_20260714" / "extracted" / "task350.onnx"
SHARED = (
    ROOT
    / "public_probe_variants"
    / "team_submission2_pending_b018_b123_b134_b285_b350_b360_v1_20260713"
    / "submission"
    / "task350.onnx"
)
OUT_DIR = ROOT / "reconstruction_candidates" / "b_task350_symbol_gate_search_v2"


def make_variant(source: onnx.ModelProto, mode: str) -> onnx.ModelProto:
    model = onnx.ModelProto.FromString(source.SerializeToString())
    node = model.graph.node[0]
    equation = next(attr.s.decode() for attr in node.attribute if attr.name == "equation")
    lhs, output = equation.split("->")
    pairs = list(zip(node.input, lhs.split(","), strict=True))

    rewritten: list[tuple[str, str]] = []
    for name, term in pairs:
        if name == "S" and term == "pq":
            if mode in {"g_outer", "g_gate_only", "no_gates", "no_gates_flip_f1"}:
                if mode in {"g_outer", "g_gate_only"}:
                    rewritten.extend([("G", "p"), ("G", "q")])
                continue
        if name == "G" and term == "x":
            if mode == "s_diagonal":
                rewritten.append(("S", "xx"))
                continue
            if mode in {"g_gate_only", "no_gates", "no_gates_flip_f1"}:
                continue
        rewritten.append((name, term))

    if mode == "no_gates_flip_f1":
        for index, initializer in enumerate(model.graph.initializer):
            if initializer.name != "F":
                continue
            array = numpy_helper.to_array(initializer).copy()
            array[1] *= -1
            model.graph.initializer[index].CopyFrom(numpy_helper.from_array(array, "F"))
            break

    used = {name for name, _ in rewritten}
    kept = [initializer for initializer in model.graph.initializer if initializer.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)

    replacement = helper.make_node(
        "Einsum",
        [name for name, _ in rewritten],
        ["output"],
        name="output",
        equation=",".join(term for _, term in rewritten) + "->" + output,
    )
    del model.graph.node[:]
    model.graph.node.append(replacement)
    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    return model


def compare(candidate: Path) -> tuple[int, int]:
    base_session = ort.InferenceSession(BASE.read_bytes(), providers=["CPUExecutionProvider"])
    candidate_session = ort.InferenceSession(candidate.read_bytes(), providers=["CPUExecutionProvider"])
    examples = json.loads((ROOT / "data" / "competition" / "task350.json").read_text())
    checked = 0
    for split in ("train", "test", "arc-gen"):
        for example in examples.get(split, []):
            pair = build_blend.convert_to_numpy(example)
            if pair is None:
                continue
            expected = base_session.run(["output"], {"input": pair["input"]})[0]
            actual = candidate_session.run(["output"], {"input": pair["input"]})[0]
            if not np.array_equal(expected > 0, actual > 0):
                return checked, checked + 1
            checked += 1
    return checked, checked


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source = onnx.load(SHARED)
    results = []
    for mode in ("s_diagonal", "g_outer", "g_gate_only", "no_gates", "no_gates_flip_f1"):
        path = OUT_DIR / f"task350_{mode}.onnx"
        try:
            onnx.save(make_variant(source, mode), path)
            right, total = compare(path)
            score = build_blend.validate_and_score((350, mode, str(path))) if right == total else None
            results.append({"mode": mode, "right": right, "total": total, "score": score})
        except Exception as exc:
            results.append({"mode": mode, "error": f"{type(exc).__name__}: {exc}"})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
