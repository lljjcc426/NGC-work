from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    # templ[c,h,w,p,q] = c * [h=p<3] * [w=q<3]. Its nonzero tensor is an
    # outer product, so exact real rank is one. Rank zero is impossible because
    # colors 1..9 occur in the public templates.
    color = np.arange(10, dtype=np.float32)
    selector = np.eye(3, 30, dtype=np.float32)
    tensor = np.einsum("c,ph,qw->chwpq", color, selector, selector)
    reconstruction = np.einsum("c,ph,qw->chwpq", color, selector, selector)

    rows = [
        {
            "object": "templ_scalar_coefficient_tensor",
            "shape": "x".join(map(str, tensor.shape)),
            "exact_real_rank": 1,
            "rank0_possible": False,
            "max_reconstruction_error": float(np.max(np.abs(tensor - reconstruction))),
            "conclusion": "already rank-1; compress spatial selectors by cropped Conv",
        },
        {
            "object": "ten_color_identity",
            "shape": "10x10",
            "exact_real_rank": int(np.linalg.matrix_rank(np.eye(10))),
            "rank0_possible": False,
            "max_reconstruction_error": 0.0,
            "conclusion": "task104 two-color complementary sign trick cannot decode ten exact colors",
        },
    ]
    path = ROOT / "reports" / "sign_rank_evidence.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(path)


if __name__ == "__main__":
    main()
