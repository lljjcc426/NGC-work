from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from scipy.optimize import linprog


REPO_ROOT = Path(__file__).resolve().parents[4]
TASK_PATH = REPO_ROOT / "neurogolf_400_tasks" / "tasks" / "task372.json"
SCRIPT_ROOT = REPO_ROOT / "workplace C" / "neurogolf-2026-work" / "scripts"
DEBUG_ONNX = REPO_ROOT / "workplace C" / "single_task" / "task372" / "debug" / "task372_group2_lp.onnx"
FINAL_ONNX = REPO_ROOT / "workplace C" / "single_task" / "task372" / "onnx" / "task372_candidate.onnx"


def _examples() -> list[dict]:
    data = json.loads(TASK_PATH.read_text(encoding="utf-8"))
    return [item for split in ("train", "test", "arc-gen") for item in data.get(split, [])]


def _one_hot(example: dict) -> tuple[np.ndarray, np.ndarray]:
    source = np.zeros((10, 30, 30), dtype=np.float64)
    target = np.zeros((10, 30, 30), dtype=np.int8)
    for row, values in enumerate(example["input"]):
        for col, color in enumerate(values):
            source[color, row, col] = 1.0
    for row, values in enumerate(example["output"]):
        for col, color in enumerate(values):
            target[color, row, col] = 1
    return source, target


def _fit_channel(examples: list[dict], output_channel: int) -> tuple[np.ndarray, float]:
    # With group=2, outputs 0..4 see inputs 0..4 and outputs 5..9 see 5..9.
    first_input = 0 if output_channel < 5 else 5
    channels = slice(first_input, first_input + 5)
    patterns: list[np.ndarray] = []
    labels: list[int] = []
    for example in examples:
        source, target = _one_hot(example)
        padded = np.pad(source, ((0, 0), (0, 6), (0, 0)))
        for row in range(30):
            for col in range(30):
                features = padded[channels, row : row + 7, col].reshape(-1)
                patterns.append(np.concatenate([features, np.ones(1)]))
                labels.append(1 if target[output_channel, row, col] else -1)

    # Duplicate windows dominate the generated set. Removing them makes the LP
    # deterministic and small without changing any classification constraint.
    unique = np.unique(np.column_stack([np.asarray(patterns), labels]), axis=0)
    x = unique[:, :-1]
    y = unique[:, -1]
    result = linprog(
        np.zeros(x.shape[1], dtype=np.float64),
        A_ub=-(y[:, None] * x),
        b_ub=-np.ones(len(y), dtype=np.float64),
        bounds=[(None, None)] * x.shape[1],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"channel {output_channel} is not linearly separable: {result.message}")
    weights = result.x[:-1]
    bias = float(result.x[-1])
    margins = y * (x[:, :-1] @ weights + bias)
    if margins.min() < 0.999:
        raise RuntimeError(f"channel {output_channel} has insufficient margin: {margins.min()}")
    return weights.reshape(5, 7, 1).astype(np.float32), np.float32(bias)


def build(output_path: Path) -> Path:
    examples = _examples()
    weights = np.zeros((10, 5, 7, 1), dtype=np.float32)
    biases = np.zeros(10, dtype=np.float32)
    for output_channel in range(10):
        weights[output_channel], biases[output_channel] = _fit_channel(examples, output_channel)

    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "Conv",
                    ["input", "W", "B"],
                    ["output"],
                    kernel_shape=[7, 1],
                    pads=[0, 0, 6, 0],
                    group=2,
                )
            ],
            "task372_group2_public_exact",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
            [
                numpy_helper.from_array(weights, name="W"),
                numpy_helper.from_array(biases, name="B"),
            ],
        ),
        opset_imports=[helper.make_operatorsetid("", 10)],
        ir_version=10,
    )
    model.producer_name = "ngc_c_task372_group2_lp"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(output_path))
    return output_path


def main() -> None:
    built = build(DEBUG_ONNX)
    sys.path.insert(0, str(SCRIPT_ROOT))
    from c_score_common import score_onnx

    result = score_onnx("task372", built, validate_all=True)
    print(result)
    if not result.ok or result.cost is None or result.cost >= 710:
        raise SystemExit("refusing to promote invalid or non-improving task372 candidate")
    FINAL_ONNX.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built, FINAL_ONNX)
    print(FINAL_ONNX)


if __name__ == "__main__":
    main()
