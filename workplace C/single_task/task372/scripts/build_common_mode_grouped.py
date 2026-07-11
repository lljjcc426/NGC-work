from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_ROOT = REPO_ROOT / "workplace C" / "neurogolf-2026-work" / "scripts"
BASE_ONNX = (
    Path("E:/kagglegolf")
    / "submissions"
    / "candidates"
    / "GOLF_20260709_101_prvsiyan_7266_72_repro"
    / "onnx"
    / "task372.onnx"
)
DEBUG_ONNX = REPO_ROOT / "workplace C" / "single_task" / "task372" / "debug" / "task372_common_mode_probe.onnx"
FINAL_ONNX = REPO_ROOT / "workplace C" / "single_task" / "task372" / "onnx" / "task372_candidate.onnx"


def _initializer(model: onnx.ModelProto, name: str) -> onnx.TensorProto:
    for init in model.graph.initializer:
        if init.name == name:
            return init
    raise KeyError(name)


def build(output_path: Path) -> Path:
    model = onnx.load(str(BASE_ONNX))
    nodes = list(model.graph.node)
    if len(nodes) != 1 or nodes[0].op_type != "Conv":
        raise RuntimeError("task372 baseline is no longer the expected single Conv graph")

    w = numpy_helper.to_array(_initializer(model, "W")).astype(np.float32)
    b = numpy_helper.to_array(_initializer(model, "B")).astype(np.float32)
    if w.shape != (10, 10, 7, 1) or b.shape != (10,):
        raise RuntimeError(f"unexpected W/B shapes: {w.shape}, {b.shape}")

    common = np.zeros((10, 7, 1), dtype=np.float32)
    for input_channel in range(10):
        off_diag = [w[out_channel, input_channel] for out_channel in range(10) if out_channel != input_channel]
        first = off_diag[0]
        if any(not np.array_equal(first, other) for other in off_diag[1:]):
            raise RuntimeError(f"input channel {input_channel} has no exact off-diagonal common kernel")
        common[input_channel] = first

    grouped = np.zeros((10, 1, 7, 1), dtype=np.float32)
    for cls in range(10):
        grouped[cls, 0] = w[cls, cls] - common[cls]

    # Prove the argmax-preserving identity on arbitrary one-hot grids before writing.
    rng = np.random.default_rng(372)
    for _ in range(64):
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        labels = rng.integers(0, 10, size=(30, 30))
        for cls in range(10):
            x[0, cls] = labels == cls
        old = np.zeros((1, 10, 30, 30), dtype=np.float32)
        new = np.zeros_like(old)
        # Vertical 7x1 Conv with pads=[0,0,6,0].
        xp = np.pad(x, ((0, 0), (0, 0), (0, 6), (0, 0)))
        for out_channel in range(10):
            for row in range(30):
                window = xp[:, :, row : row + 7, :]
                old[:, out_channel, row] = (window * w[out_channel, :, :, :]).sum(axis=(1, 2))
                new[:, out_channel, row] = (window[:, out_channel : out_channel + 1] * grouped[out_channel]).sum(axis=(1, 2))
            old[:, out_channel] += b[out_channel]
            new[:, out_channel] += b[out_channel]
        if not np.array_equal(old.argmax(axis=1), new.argmax(axis=1)):
            raise RuntimeError("common-mode grouped Conv changed argmax in synthetic probe")

    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv",
                ["input", "W_grouped", "B"],
                ["output"],
                name="output",
                kernel_shape=[7, 1],
                pads=[0, 0, 6, 0],
                group=10,
            )
        ],
        model.graph.name,
        model.graph.input,
        model.graph.output,
        [
            numpy_helper.from_array(grouped, name="W_grouped"),
            numpy_helper.from_array(b, name="B"),
        ],
        value_info=model.graph.value_info,
    )
    candidate = helper.make_model(graph, opset_imports=model.opset_import, ir_version=model.ir_version)
    candidate.producer_name = "ngc_c_task372_common_mode_grouped"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(candidate)
    onnx.save(candidate, str(output_path))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()

    built = build(DEBUG_ONNX)
    print(built)
    sys.path.insert(0, str(SCRIPT_ROOT))
    from c_score_common import score_onnx

    result = score_onnx("task372", built, validate_all=True)
    print(
        {
            "ok": result.ok,
            "examples": f"{result.examples_passed}/{result.examples_checked}",
            "memory": result.memory,
            "params": result.params,
            "cost": result.cost,
            "points": result.points,
            "error": result.error,
        }
    )
    if args.promote:
        if not result.ok or result.cost is None or result.cost >= 710:
            raise SystemExit("refusing to promote non-improving task372 candidate")
        FINAL_ONNX.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built, FINAL_ONNX)
        print(FINAL_ONNX)


if __name__ == "__main__":
    main()
