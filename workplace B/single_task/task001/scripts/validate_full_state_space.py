from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = REPO_ROOT / "workplace C" / "neurogolf-2026-work" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from c_score_common import load_official_utils  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_grid(mask: np.ndarray, color: int) -> list[list[int]]:
    return (np.kron(mask, mask) * color).astype(np.int64).tolist()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()

    model = onnx.load(args.model)
    sanitized = load_official_utils().sanitize_model(model)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    utils = load_official_utils()
    checked = 0
    for bits in range(1 << 9):
        mask = np.asarray([(bits >> i) & 1 for i in range(9)], dtype=np.int64).reshape(3, 3)
        for color in range(1, 10):
            grid = (mask * color).tolist()
            example = {"input": grid, "output": expected_grid(mask, color)}
            arrays = utils.convert_to_numpy(example)
            output = utils.run_network(session, arrays["input"])
            if not np.array_equal(output, arrays["output"]):
                print({"status": "failed", "bits": bits, "color": color, "checked": checked})
                return 1
            checked += 1
    print(
        {
            "status": "passed",
            "states_checked": checked,
            "binary_patterns": 1 << 9,
            "colors": 9,
            "candidate_sha256": sha256(args.model),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
