from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


TASK_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TASK_DIR.parents[2]
SOURCE = Path(
    r"E:/kagglegolf/submissions/candidates/"
    r"GOLF_20260711_096_v95_plus_4_compact/onnx/task077.onnx"
)
TASK_JSON = REPO_ROOT / "neurogolf_400_tasks" / "tasks" / "task077.json"
UTILS = Path(r"E:/kagglegolf/data/raw/neurogolf-2026/neurogolf_utils/neurogolf_utils.py")
DEBUG = TASK_DIR / "debug" / "dilation_search"
REPORT = TASK_DIR / "reports" / "dilation_search.csv"


def load_utils():
    spec = importlib.util.spec_from_file_location("task077_utils", UTILS)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {UTILS}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._NEUROGOLF_DIR = "E:/kagglegolf/data/raw/neurogolf-2026/"
    return module


def examples() -> list[dict[str, np.ndarray]]:
    utils = load_utils()
    payload = json.loads(TASK_JSON.read_text(encoding="utf-8"))
    return [
        utils.convert_to_numpy(example)
        for split in ("train", "test", "arc-gen")
        for example in payload[split]
    ]


def set_pool(node: onnx.NodeProto, kernel: int, dilation: int) -> None:
    del node.attribute[:]
    radius = (kernel - 1) * dilation // 2
    node.attribute.extend(
        [
            onnx.helper.make_attribute("kernel_shape", [1, kernel]),
            onnx.helper.make_attribute("pads", [0, radius, 0, radius]),
            onnx.helper.make_attribute("dilations", [1, dilation]),
            onnx.helper.make_attribute("strides", [1, 1]),
        ]
    )


def build(kernel1: int, dilation1: int, kernel2: int, dilation2: int) -> onnx.ModelProto:
    model = onnx.load(str(SOURCE))
    nodes = list(model.graph.node)
    if [nodes[index].op_type for index in (4, 6, 8, 9)] != ["MaxPool", "MaxPool", "MaxPool", "Min"]:
        raise RuntimeError("unexpected task077 graph")
    set_pool(nodes[4], kernel1, dilation1)
    set_pool(nodes[6], kernel2, dilation2)
    nodes[7].output[0] = "F3"
    del model.graph.node[:]
    model.graph.node.extend(nodes[:8] + nodes[10:])
    onnx.checker.check_model(model, full_check=True)
    return model


def validate(model: onnx.ModelProto, rows: list[dict[str, np.ndarray]]) -> tuple[int, int]:
    session = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=ort.SessionOptions(),
        providers=["CPUExecutionProvider"],
    )
    passed = 0
    mismatch = 0
    for row in rows:
        output = (session.run(["output"], {"input": row["input"]})[0] > 0).astype(np.float32)
        errors = int(np.sum(output != row["output"]))
        passed += errors == 0
        mismatch += errors
    return passed, mismatch


def main() -> int:
    DEBUG.mkdir(parents=True, exist_ok=True)
    rows = examples()
    # ORT requires each explicit pad to be smaller than the kernel. Dilation 2
    # is the largest useful legal sparse support for these odd kernels.
    schedules = [(kernel, dilation) for kernel in (3, 5, 7, 9, 11, 13) for dilation in (1, 2)]
    results: list[dict[str, int | str]] = []
    best: tuple[int, int, str, onnx.ModelProto] | None = None
    for kernel1, dilation1 in schedules:
        radius1 = (kernel1 - 1) * dilation1 // 2
        for kernel2, dilation2 in schedules:
            radius2 = (kernel2 - 1) * dilation2 // 2
            if not 5 <= radius1 + radius2 <= 9:
                continue
            name = f"k{kernel1}d{dilation1}_k{kernel2}d{dilation2}"
            model = build(kernel1, dilation1, kernel2, dilation2)
            passed, mismatch = validate(model, rows)
            results.append(
                {
                    "name": name,
                    "kernel1": kernel1,
                    "dilation1": dilation1,
                    "kernel2": kernel2,
                    "dilation2": dilation2,
                    "radius_sum": radius1 + radius2,
                    "passed": passed,
                    "mismatch_cells": mismatch,
                }
            )
            rank = (-passed, mismatch, name, model)
            if best is None or rank[:2] < best[:2]:
                best = rank
                print(name, passed, mismatch, flush=True)
            if passed == len(rows):
                output = TASK_DIR / "onnx" / "task077_dilated_two_round.onnx"
                onnx.save(model, str(output))
                print(f"VALID {name}: {output}")
                break
        if results and results[-1]["passed"] == len(rows):
            break
    with REPORT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda row: (-int(row["passed"]), int(row["mismatch_cells"]))))
    print(f"searched={len(results)} best={REPORT}")
    return 0 if any(row["passed"] == len(rows) for row in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
