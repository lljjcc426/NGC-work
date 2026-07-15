from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import build_blend  # noqa: E402


MODEL = ROOT / "reconstruction_candidates" / "b_task344_rank2_joint_v3" / "task344.onnx"


def main() -> None:
    data = json.loads((ROOT / "data" / "competition" / "task344.json").read_text())
    session = ort.InferenceSession(MODEL.read_bytes(), providers=["CPUExecutionProvider"])
    total = 0
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(data.get(split, [])):
            pair = build_blend.convert_to_numpy(example)
            if pair is None:
                continue
            logits = session.run(["output"], {"input": pair["input"]})[0]
            wrong = np.argwhere((logits > 0) != pair["output"])
            for batch, channel, row, col in wrong:
                grid = example["input"]
                neighborhood = []
                for rr in range(row - 1, row + 2):
                    cells = []
                    for cc in range(col - 1, col + 2):
                        cells.append(grid[rr][cc] if 0 <= rr < len(grid) and 0 <= cc < len(grid[0]) else -1)
                    neighborhood.append(cells)
                print(
                    json.dumps(
                        {
                            "split": split,
                            "example": index,
                            "channel": int(channel),
                            "row": int(row),
                            "col": int(col),
                            "logit": float(logits[batch, channel, row, col]),
                            "target": float(pair["output"][batch, channel, row, col]),
                            "neighborhood": neighborhood,
                        }
                    )
                )
                total += 1
    print(json.dumps({"total": total}))


if __name__ == "__main__":
    main()
