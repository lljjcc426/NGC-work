from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from numpy.lib.stride_tricks import sliding_window_view
from onnx import TensorProto, helper


HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
TASK_ROOT = REPO / "neurogolf_400_tasks" / "tasks"
DEFAULT_OUTPUT = REPO / "workplace C" / "artifacts" / "zero_cost_unary_search"


def encode(grid: list[list[int]]) -> np.ndarray:
    value = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for row, cells in enumerate(grid):
        for col, color in enumerate(cells):
            value[0, color, row, col] = 1.0
    return value


def load_pairs(path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = []
    for split in ("train", "test", "arc-gen"):
        for item in payload.get(split, []):
            grids = (item["input"], item["output"])
            if any(max(len(g), len(g[0]) if g else 0) > 30 for g in grids):
                continue
            result.append((encode(grids[0]), encode(grids[1])))
    return result


def shift(source: np.ndarray, dr: int, dc: int) -> np.ndarray:
    result = np.zeros_like(source, dtype=bool)
    src_r0, src_r1 = max(0, -dr), min(30, 30 - dr)
    src_c0, src_c1 = max(0, -dc), min(30, 30 - dc)
    dst_r0, dst_r1 = src_r0 + dr, src_r1 + dr
    dst_c0, dst_c1 = src_c0 + dc, src_c1 + dc
    result[:, :, dst_r0:dst_r1, dst_c0:dst_c1] = (
        source[:, :, src_r0:src_r1, src_c0:src_c1] > 0
    )
    return result


def pool(source: np.ndarray, kh: int, kw: int) -> np.ndarray:
    top, bottom = kh // 2, kh - 1 - kh // 2
    left, right = kw // 2, kw - 1 - kw // 2
    padded = np.pad(source, ((0, 0), (0, 0), (top, bottom), (left, right)))
    windows = sliding_window_view(padded, (kh, kw), axis=(-2, -1))
    return windows.max(axis=(-1, -2)) > 0


def trilu(source: np.ndarray, upper: bool) -> np.ndarray:
    mask = np.triu(np.ones((30, 30), dtype=bool)) if upper else np.tril(
        np.ones((30, 30), dtype=bool)
    )
    return (source > 0) & mask


def templates() -> list[tuple[str, tuple[int, ...]]]:
    result: list[tuple[str, tuple[int, ...]]] = []
    for dr in range(-10, 11):
        for dc in range(-10, 11):
            if dr or dc:
                result.append(("pad_shift", (dr, dc)))
    for size in range(2, 10):
        result.extend(
            [
                ("maxpool", (1, size)),
                ("maxpool", (size, 1)),
                ("maxpool", (size, size)),
            ]
        )
    result.extend([("trilu", (1,)), ("trilu", (0,))])
    return result


def predict(source: np.ndarray, kind: str, attrs: tuple[int, ...]) -> np.ndarray:
    if kind == "pad_shift":
        return shift(source, *attrs)
    if kind == "maxpool":
        return pool(source, *attrs)
    if kind == "trilu":
        return trilu(source, bool(attrs[0]))
    raise ValueError(kind)


def matches(pairs: list[tuple[np.ndarray, np.ndarray]], kind: str, attrs: tuple[int, ...]) -> bool:
    return all(np.array_equal(predict(source, kind, attrs), target > 0) for source, target in pairs)


def build(kind: str, attrs: tuple[int, ...], path: Path) -> None:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    if kind == "pad_shift":
        dr, dc = attrs
        node = helper.make_node(
            "Pad",
            ["input"],
            ["output"],
            mode="constant",
            pads=[0, 0, dr, dc, 0, 0, -dr, -dc],
            value=0.0,
        )
        opset = 10
    elif kind == "maxpool":
        kh, kw = attrs
        node = helper.make_node(
            "MaxPool",
            ["input"],
            ["output"],
            kernel_shape=[kh, kw],
            pads=[kh // 2, kw // 2, kh - 1 - kh // 2, kw - 1 - kw // 2],
            strides=[1, 1],
        )
        opset = 18
    elif kind == "trilu":
        node = helper.make_node("Trilu", ["input"], ["output"], upper=attrs[0])
        opset = 18
    else:
        raise ValueError(kind)
    graph = helper.make_graph([node], "zero_cost_unary", [x], [y])
    model = helper.make_model(
        graph, ir_version=8, opset_imports=[helper.make_opsetid("", opset)]
    )
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    hits = []
    candidates = templates()
    for task_path in sorted(TASK_ROOT.glob("task*.json")):
        pairs = load_pairs(task_path)
        for kind, attrs in candidates:
            if not matches(pairs, kind, attrs):
                continue
            name = kind + "_" + "_".join(map(str, attrs))
            output = args.output_dir / task_path.stem / f"{name}.onnx"
            build(kind, attrs, output)
            hit = {"task": task_path.stem, "kind": kind, "attrs": attrs, "path": str(output)}
            hits.append(hit)
            print(json.dumps(hit), flush=True)
    print(json.dumps({"templates": len(candidates), "hits": hits}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
