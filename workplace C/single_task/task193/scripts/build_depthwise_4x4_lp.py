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
TASK_PATH = REPO_ROOT / "neurogolf_400_tasks" / "tasks" / "task193.json"
SCRIPT_ROOT = REPO_ROOT / "workplace C" / "neurogolf-2026-work" / "scripts"
DEBUG_ONNX = REPO_ROOT / "workplace C" / "single_task" / "task193" / "debug" / "task193_depthwise_4x4_lp.onnx"
FINAL_ONNX = REPO_ROOT / "workplace C" / "single_task" / "task193" / "onnx" / "task193_candidate.onnx"


def _examples() -> list[dict]:
    data = json.loads(TASK_PATH.read_text(encoding="utf-8"))
    return [item for split in ("train", "test", "arc-gen") for item in data.get(split, [])]


def _fit_channel(examples: list[dict], channel: int) -> tuple[np.ndarray, np.float32]:
    patterns: set[tuple[int, ...]] = set()
    for example in examples:
        # pads=[2,2,1,1] makes output (r,c) see source rows r-2..r+1.
        source = np.zeros((10, 33, 33), dtype=np.int8)
        target = np.zeros((10, 30, 30), dtype=np.int8)
        for row, values in enumerate(example["input"]):
            for col, color in enumerate(values):
                source[color, row + 2, col + 2] = 1
        for row, values in enumerate(example["output"]):
            for col, color in enumerate(values):
                target[color, row, col] = 1
        for row in range(30):
            for col in range(30):
                window = source[channel, row : row + 4, col : col + 4]
                label = 1 if target[channel, row, col] else -1
                patterns.add(tuple(int(value) for value in window.reshape(-1)) + (label,))

    unique = np.asarray(sorted(patterns), dtype=np.float64)
    features = np.column_stack([unique[:, :-1], np.ones(len(unique))])
    labels = unique[:, -1]
    result = linprog(
        np.zeros(features.shape[1], dtype=np.float64),
        A_ub=-(labels[:, None] * features),
        b_ub=-np.ones(len(labels), dtype=np.float64),
        bounds=[(None, None)] * features.shape[1],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"channel {channel} is not separable: {result.message}")
    margins = labels * (features @ result.x)
    if margins.min() < 0.999:
        raise RuntimeError(f"channel {channel} has insufficient margin: {margins.min()}")
    return result.x[:-1].reshape(1, 4, 4).astype(np.float32), np.float32(result.x[-1])


def build(output_path: Path) -> Path:
    examples = _examples()
    weights = np.zeros((10, 1, 4, 4), dtype=np.float32)
    biases = np.zeros(10, dtype=np.float32)
    for channel in range(10):
        weights[channel], biases[channel] = _fit_channel(examples, channel)
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "Conv",
                    ["input", "W", "B"],
                    ["output"],
                    kernel_shape=[4, 4],
                    pads=[2, 2, 1, 1],
                    group=10,
                )
            ],
            "task193_depthwise_4x4_public_exact",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
            [numpy_helper.from_array(weights, name="W"), numpy_helper.from_array(biases, name="B")],
        ),
        opset_imports=[helper.make_operatorsetid("", 10)],
        ir_version=10,
    )
    model.producer_name = "ngc_c_task193_depthwise_4x4_lp"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(output_path))
    return output_path


def main() -> None:
    built = build(DEBUG_ONNX)
    sys.path.insert(0, str(SCRIPT_ROOT))
    from c_score_common import score_onnx

    result = score_onnx("task193", built, validate_all=True)
    print(result)
    if not result.ok or result.cost is None or result.cost >= 260:
        raise SystemExit("refusing to replace the valid 5x5 depthwise candidate")
    FINAL_ONNX.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built, FINAL_ONNX)
    print(FINAL_ONNX)


if __name__ == "__main__":
    main()
