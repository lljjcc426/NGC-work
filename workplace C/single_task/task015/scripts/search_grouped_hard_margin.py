from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import helper, numpy_helper
from sklearn.svm import LinearSVC


TASK_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TASK_DIR.parents[2]
TASK_JSON = REPO_ROOT / "neurogolf_400_tasks" / "tasks" / "task015.json"
UTILS = Path(r"E:/kagglegolf/data/raw/neurogolf-2026/neurogolf_utils/neurogolf_utils.py")
SOURCE = Path(
    r"E:/kagglegolf/submissions/candidates/"
    r"GOLF_20260711_096_v95_plus_4_compact/onnx/task015.onnx"
)
REPORT = TASK_DIR / "reports" / "hard_margin_search.csv"


def load_utils():
    spec = importlib.util.spec_from_file_location("task015_utils", UTILS)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {UTILS}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._NEUROGOLF_DIR = "E:/kagglegolf/data/raw/neurogolf-2026/"
    return module


def load_arrays() -> tuple[np.ndarray, np.ndarray]:
    utils = load_utils()
    payload = json.loads(TASK_JSON.read_text(encoding="utf-8"))
    rows = [
        utils.convert_to_numpy(example)
        for split in ("train", "test", "arc-gen")
        for example in payload[split]
    ]
    return np.concatenate([row["input"] for row in rows]), np.concatenate([row["output"] for row in rows])


def windows(inputs: np.ndarray, channels: slice, kernel: int, top: int, left: int) -> np.ndarray:
    bottom = kernel - 1 - top
    right = kernel - 1 - left
    tensor = torch.from_numpy(inputs[:, channels])
    unfolded = F.unfold(F.pad(tensor, (left, right, top, bottom)), kernel_size=(kernel, kernel))
    return unfolded.transpose(1, 2).reshape(-1, unfolded.shape[1]).numpy().astype(np.uint8)


def deduplicate(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    packed = np.packbits(x, axis=1)
    unique, first, inverse = np.unique(packed, axis=0, return_index=True, return_inverse=True)
    del unique
    y_min = np.full(len(first), 2, dtype=np.int8)
    y_max = np.full(len(first), -1, dtype=np.int8)
    np.minimum.at(y_min, inverse, y)
    np.maximum.at(y_max, inverse, y)
    conflicts = int(np.sum(y_min != y_max))
    return x[first], y[first], conflicts


def fit_channel(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray | None, float | None, int, int]:
    unique_x, unique_y, conflicts = deduplicate(x, y)
    if conflicts:
        return None, None, conflicts, -1
    classes = np.unique(unique_y)
    if len(classes) == 1:
        bias = 1.0 if int(classes[0]) else -1.0
        return np.zeros(unique_x.shape[1], dtype=np.float32), bias, 0, 0
    classifier = LinearSVC(C=1e6, dual="auto", tol=1e-10, max_iter=200000)
    classifier.fit(unique_x, unique_y)
    errors = int(np.sum(classifier.predict(unique_x) != unique_y))
    if errors:
        return None, None, 0, errors
    return classifier.coef_[0].astype(np.float32), float(classifier.intercept_[0]), 0, 0


def group_channels(group: int, output_channel: int) -> slice:
    outputs_per_group = 10 // group
    inputs_per_group = 10 // group
    group_index = output_channel // outputs_per_group
    return slice(group_index * inputs_per_group, (group_index + 1) * inputs_per_group)


def build_candidate(
    group: int,
    kernel: int,
    top: int,
    left: int,
    weights: list[np.ndarray],
    biases: list[float],
    output_path: Path,
) -> Path:
    input_channels = 10 // group
    weight = np.stack(weights).reshape(10, input_channels, kernel, kernel).astype(np.float32)
    bias = np.asarray(biases, dtype=np.float32)
    model = onnx.load(str(SOURCE))
    del model.graph.node[:]
    del model.graph.initializer[:]
    model.graph.initializer.extend(
        [numpy_helper.from_array(weight, name="W_grouped"), numpy_helper.from_array(bias, name="B_grouped")]
    )
    model.graph.node.append(
        helper.make_node(
            "Conv",
            ["input", "W_grouped", "B_grouped"],
            ["output"],
            group=group,
            kernel_shape=[kernel, kernel],
            pads=[top, left, kernel - 1 - top, kernel - 1 - left],
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(output_path))
    return output_path


def main() -> int:
    inputs, outputs = load_arrays()
    results: list[dict[str, int | str]] = []
    candidate_count = 0
    for group, kernel in ((2, 4), (2, 5), (5, 4), (5, 5)):
        cache: dict[tuple[int, int, int, int], np.ndarray] = {}
        for top in range(kernel):
            for left in range(kernel):
                fitted_weights: list[np.ndarray] = []
                fitted_biases: list[float] = []
                failed_channels: list[int] = []
                conflict_total = 0
                margin_errors = 0
                for output_channel in range(10):
                    channels = group_channels(group, output_channel)
                    key = (channels.start or 0, channels.stop or 10, top, left)
                    if key not in cache:
                        cache[key] = windows(inputs, channels, kernel, top, left)
                    x = cache[key]
                    y = outputs[:, output_channel].reshape(-1).astype(np.int8)
                    weight, bias, conflicts, errors = fit_channel(x, y)
                    conflict_total += conflicts
                    margin_errors += max(0, errors)
                    if weight is None or bias is None:
                        failed_channels.append(output_channel)
                        fitted_weights.append(np.zeros(x.shape[1], dtype=np.float32))
                        fitted_biases.append(-1.0)
                    else:
                        fitted_weights.append(weight)
                        fitted_biases.append(bias)
                status = "separable" if not failed_channels else "failed"
                name = f"g{group}_k{kernel}_p{top}{left}"
                results.append(
                    {
                        "name": name,
                        "group": group,
                        "kernel": kernel,
                        "pad_top": top,
                        "pad_left": left,
                        "failed_channels": " ".join(map(str, failed_channels)),
                        "conflicting_windows": conflict_total,
                        "margin_errors": margin_errors,
                        "status": status,
                        "parameter_cost": 10 * (10 // group) * kernel * kernel + 10,
                    }
                )
                if not failed_channels:
                    output_path = TASK_DIR / "onnx" / f"task015_{name}.onnx"
                    build_candidate(group, kernel, top, left, fitted_weights, fitted_biases, output_path)
                    candidate_count += 1
                    print(f"SEPARABLE {name}: {output_path}", flush=True)
                elif len(failed_channels) <= 3:
                    print(name, "failed", failed_channels, "conflicts", conflict_total, flush=True)
    with REPORT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"configurations={len(results)} separable={candidate_count} report={REPORT}")
    return 0 if candidate_count else 2


if __name__ == "__main__":
    raise SystemExit(main())
