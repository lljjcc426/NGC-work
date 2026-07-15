from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


REPO = Path(__file__).resolve().parents[4]


def _session(path: Path, expose_selector: bool = False) -> ort.InferenceSession:
    model = onnx.shape_inference.infer_shapes(
        onnx.load(path), strict_mode=True, data_prop=True
    )
    if expose_selector:
        value_info = next(value for value in model.graph.value_info if value.name == "pbm")
        model.graph.output.append(value_info)
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


def _run(
    parent: ort.InferenceSession,
    candidate: ort.InferenceSession,
    grid: np.ndarray,
) -> tuple[bool, int]:
    encoded = _encode(grid)
    parent_output, selector = parent.run(
        ["output", "pbm"], {parent.get_inputs()[0].name: encoded}
    )
    candidate_output = candidate.run(
        None, {candidate.get_inputs()[0].name: encoded}
    )[0]
    return np.array_equal(parent_output, candidate_output), int(np.count_nonzero(selector))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify task133 scalar selection under nonzero color permutations."
    )
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1332026)
    args = parser.parse_args()

    parent = _session(args.parent, expose_selector=True)
    candidate = _session(args.candidate)
    payload = json.loads(
        (REPO / "neurogolf_400_tasks" / "tasks" / "task133.json").read_text(
            encoding="utf-8"
        )
    )
    official = [
        np.asarray(example["input"], dtype=np.int64)
        for split in ("train", "test", "arc-gen")
        for example in payload[split]
    ]
    official_passed = 0
    official_one_hot = 0
    for grid in official:
        equal, selected = _run(parent, candidate, grid)
        official_passed += int(equal)
        official_one_hot += int(selected == 1)

    rng = np.random.default_rng(args.seed)
    generated_passed = 0
    generated_one_hot = 0
    for trial in range(args.trials):
        mapping = np.arange(10, dtype=np.int64)
        mapping[1:] = rng.permutation(np.arange(1, 10))
        grid = mapping[official[trial % len(official)]]
        equal, selected = _run(parent, candidate, grid)
        if not equal or selected != 1:
            print(
                json.dumps(
                    {
                        "status": "mismatch",
                        "trial": trial,
                        "equal": equal,
                        "selector_nonzero": selected,
                    },
                    separators=(",", ":"),
                )
            )
            return 1
        generated_passed += 1
        generated_one_hot += 1

    result = {
        "status": "passed",
        "official_checked": len(official),
        "official_passed": official_passed,
        "official_selector_one_hot": official_one_hot,
        "permutation_trials": args.trials,
        "permutation_passed": generated_passed,
        "permutation_selector_one_hot": generated_one_hot,
        "parent_sha256": hashlib.sha256(args.parent.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
    }
    print(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
