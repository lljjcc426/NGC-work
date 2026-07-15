from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


REPO = Path(__file__).resolve().parents[4]


def _session(path: Path, expose_pick: bool = False) -> ort.InferenceSession:
    model = onnx.load(path)
    if expose_pick:
        produced = {name for node in model.graph.node for name in node.output}
        if "pick" in produced:
            model.graph.output.append(
                helper.make_tensor_value_info("pick", TensorProto.BOOL, [3, 1])
            )
        elif "pick3" in produced:
            model.graph.node.append(
                helper.make_node("Identity", ["pick3"], ["pick"], name="expose_pick3")
            )
            model.graph.output.append(
                helper.make_tensor_value_info("pick", TensorProto.UINT8, [3])
            )
        else:
            raise RuntimeError(f"no task174 selector in {path}")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def _encode(grid: np.ndarray) -> np.ndarray:
    encoded = np.zeros((1, 10, 30, 30), dtype=np.float32)
    rows, columns = np.indices(grid.shape)
    encoded[0, grid, rows, columns] = 1.0
    return encoded


def _symmetric_mask(rng: np.random.Generator) -> np.ndarray:
    height = int(rng.integers(1, 6))
    width = int(rng.integers(1, 6))
    kind = int(rng.integers(4))
    mask = np.zeros((height, width), dtype=bool)
    if kind == 0:
        mask[:] = True
    elif kind == 1:
        mask[0] = True
        mask[:, (width - 1) // 2] = True
        mask[:, width // 2] = True
    elif kind == 2:
        mask[0] = True
        mask[-1] = True
        mask[:, 0] = True
        mask[:, -1] = True
    else:
        center_row = (height - 1) / 2
        center_column = (width - 1) / 2
        radius = max(center_row + center_column, 1)
        for row in range(height):
            for column in range(width):
                mask[row, column] = (
                    abs(row - center_row) + abs(column - center_column)
                    <= radius * 0.72 + 0.5
                )
    mask |= mask[:, ::-1]
    mask[height // 2, width // 2] = True
    return mask


def _asymmetric_mask(rng: np.random.Generator) -> np.ndarray:
    height = int(rng.integers(2, 6))
    width = int(rng.integers(2, 6))
    mask = np.zeros((height, width), dtype=bool)
    orientation = int(rng.integers(4))
    if orientation == 0:
        mask[:, 0] = True
        mask[-1] = True
    elif orientation == 1:
        mask[:, -1] = True
        mask[0] = True
    elif orientation == 2:
        mask[0] = True
        mask[:, 0] = True
        mask[-1, -1] = True
    else:
        mask[-1] = True
        mask[:, -1] = True
        mask[0, 0] = True
    if np.array_equal(mask, mask[:, ::-1]):
        mask[0, 0] = True
        mask[0, -1] = False
    return mask


def _place_objects(
    rng: np.random.Generator, masks: list[np.ndarray], colors: np.ndarray
) -> np.ndarray | None:
    for _ in range(300):
        grid = np.zeros((10, 10), dtype=np.int64)
        occupied = np.zeros((10, 10), dtype=bool)
        success = True
        for index in rng.permutation(len(masks)):
            mask = masks[int(index)]
            height, width = mask.shape
            placed = False
            for _ in range(100):
                top = int(rng.integers(0, 11 - height))
                left = int(rng.integers(0, 11 - width))
                near = np.zeros_like(occupied)
                rows, columns = np.where(occupied)
                for row, column in zip(rows, columns):
                    near[
                        max(0, row - 1) : min(10, row + 2),
                        max(0, column - 1) : min(10, column + 2),
                    ] = True
                if np.any(near[top : top + height, left : left + width] & mask):
                    continue
                view = grid[top : top + height, left : left + width]
                view[mask] = int(colors[int(index)])
                occupied[top : top + height, left : left + width] |= mask
                placed = True
                break
            if not placed:
                success = False
                break
        if success:
            return grid
    return None


def _compare(
    parent_session: ort.InferenceSession,
    candidate_session: ort.InferenceSession,
    grid: np.ndarray,
) -> tuple[bool, int]:
    encoded = _encode(grid)
    parent_output, pick = parent_session.run(
        ["output", "pick"], {parent_session.get_inputs()[0].name: encoded}
    )
    candidate_output = candidate_session.run(
        None, {candidate_session.get_inputs()[0].name: encoded}
    )[0]
    return np.array_equal(parent_output, candidate_output), int(np.count_nonzero(pick))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fuzz task174 on unseen three-object layouts with one symmetric object."
    )
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1742026)
    args = parser.parse_args()

    parent_session = _session(args.parent, expose_pick=True)
    candidate_session = _session(args.candidate)
    task = json.loads(
        (REPO / "neurogolf_400_tasks" / "tasks" / "task174.json").read_text(
            encoding="utf-8"
        )
    )
    official = [
        np.asarray(example["input"], dtype=np.int64)
        for split in ("train", "test", "arc-gen")
        for example in task[split]
    ]
    official_passed = 0
    official_one_hot = 0
    for grid in official:
        equal, pick_count = _compare(parent_session, candidate_session, grid)
        official_passed += int(equal)
        official_one_hot += int(pick_count == 1)

    rng = np.random.default_rng(args.seed)
    generated_passed = 0
    attempts = 0
    selector_rejected = 0
    placement_rejected = 0
    while generated_passed < args.trials and attempts < args.trials * 20:
        attempts += 1
        colors = rng.choice(np.arange(1, 10), 3, replace=False)
        masks = [_symmetric_mask(rng), _asymmetric_mask(rng), _asymmetric_mask(rng)]
        grid = _place_objects(rng, masks, colors)
        if grid is None:
            placement_rejected += 1
            continue
        equal, pick_count = _compare(parent_session, candidate_session, grid)
        if pick_count != 1:
            selector_rejected += 1
            continue
        if not equal:
            print(
                json.dumps(
                    {
                        "status": "mismatch",
                        "trial": generated_passed,
                        "attempt": attempts,
                        "grid": grid.tolist(),
                    },
                    separators=(",", ":"),
                )
            )
            return 1
        generated_passed += 1

    result = {
        "status": "passed"
        if official_passed == len(official)
        and official_one_hot == len(official)
        and generated_passed == args.trials
        else "failed",
        "official_checked": len(official),
        "official_passed": official_passed,
        "official_selector_one_hot": official_one_hot,
        "generated_requested": args.trials,
        "generated_passed": generated_passed,
        "attempts": attempts,
        "selector_rejected": selector_rejected,
        "placement_rejected": placement_rejected,
        "parent_sha256": hashlib.sha256(args.parent.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
    }
    print(json.dumps(result, separators=(",", ":")))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
