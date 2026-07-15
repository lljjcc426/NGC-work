from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper


REPO_ROOT = Path(__file__).resolve().parents[4]
TASK_ROOT = Path(__file__).resolve().parents[1]
PARENT = (
    REPO_ROOT
    / "workplace C"
    / "artifacts"
    / "full400_round36_public_source_safe37"
    / "onnx"
    / "task001.onnx"
)
OUTPUT = TASK_ROOT / "debug" / "task001_sparse_parent_cost37_unscoreable.onnx"
REPORT = TASK_ROOT / "reports" / "task001_sparse_parent_probe.json"
sys.path.insert(0, str(REPO_ROOT / "workplace C" / "neurogolf-2026-work" / "scripts"))

from c_score_common import load_official_utils, score_onnx  # noqa: E402


def attempt(callable_object) -> dict[str, object]:
    try:
        callable_object()
        return {"ok": True, "error": ""}
    except Exception as exc:  # The exact checker error is the evidence.
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def build() -> onnx.ModelProto:
    model = onnx.load(PARENT)
    dense = next(value for value in model.graph.initializer if value.name == "m")
    array = numpy_helper.to_array(dense)
    coordinates = np.argwhere(array != 0).astype(np.int64)
    values = array[tuple(coordinates.T)].astype(np.float32)
    sparse = helper.make_sparse_tensor(
        # sanitize_model renames the dense `v` initializer to safe_name_0 but
        # skips sparse initializers. Pre-naming this operand isolates the later
        # strict shape-inference failure from that sanitizer naming bug.
        numpy_helper.from_array(values, name="safe_name_1"),
        numpy_helper.from_array(coordinates, name="m_indices"),
        list(array.shape),
    )
    kept = [value for value in model.graph.initializer if value.name != "m"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.sparse_initializer.append(sparse)
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == "m":
                node.input[index] = "safe_name_1"
    model.graph.name = "task001_sparse_parent_cost37_unscoreable"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, OUTPUT)
    return model


def direct_ort_bundled(model: onnx.ModelProto) -> dict[str, int]:
    payload = json.loads(
        (REPO_ROOT / "neurogolf_400_tasks" / "tasks" / "task001.json").read_text(
            encoding="utf-8"
        )
    )
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    utils = load_official_utils()
    checked = 0
    passed = 0
    for split in ("train", "test", "arc-gen"):
        for example in payload[split]:
            arrays = utils.convert_to_numpy(example)
            actual = (session.run(["output"], {"input": arrays["input"]})[0] > 0).astype(
                np.float32
            )
            checked += 1
            passed += int(np.array_equal(actual, arrays["output"]))
    return {"checked": checked, "passed": passed}


def main() -> int:
    model = build()
    sparse_values = int(model.graph.sparse_initializer[0].values.dims[0])
    result: dict[str, object] = {
        "path": str(OUTPUT),
        "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "theoretical_params": 10 + sparse_values,
        "theoretical_memory": 0,
        "theoretical_cost": 10 + sparse_values,
        "direct_ort": direct_ort_bundled(model),
        "checker_default": attempt(lambda: onnx.checker.check_model(model)),
        "checker_full": attempt(lambda: onnx.checker.check_model(model, full_check=True)),
        "shape_inference_strict": attempt(
            lambda: onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        ),
    }
    score = score_onnx("task001", OUTPUT, validate_all=True)
    result["official_style_score"] = vars(score)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
