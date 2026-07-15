from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import build_blend  # noqa: E402
import train_task344_rank2_joint as joint  # noqa: E402


SOURCE = ROOT / "reconstruction_candidates" / "b_task344_rank2_joint_v3" / "task344.onnx"
OUT = ROOT / "reconstruction_candidates" / "b_task344_rank2_hard_v4" / "task344.onnx"


def save_candidate(parameters: list[torch.Tensor]) -> None:
    previous = joint.OUT
    joint.OUT = OUT
    try:
        joint.save_candidate(*(item.detach().numpy() for item in parameters))
    finally:
        joint.OUT = previous


def main() -> None:
    torch.manual_seed(3444)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    arrays = {
        item.name: numpy_helper.to_array(item).astype(np.float32)
        for item in onnx.load(SOURCE).graph.initializer
    }
    parameters = [
        torch.nn.Parameter(torch.from_numpy(arrays[name].copy()))
        for name in ("M_left", "M_right", "U", "G", "A")
    ]
    anchors = [item.detach().clone() for item in parameters]
    inputs, targets = joint.load_examples()
    signs = targets.float().mul(2).sub(1)
    optimizer = torch.optim.Adam(parameters, lr=0.0002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000, eta_min=0.000005)

    best_wrong = targets.numel()
    best_state: tuple[np.ndarray, ...] | None = None
    for step in range(1, 5001):
        output = joint.infer(inputs, *parameters)
        signed = signs * output
        hard = torch.topk(F.softplus(1.0 - signed).flatten(), k=512).values.mean()
        anchor = sum(F.mse_loss(item, reference) for item, reference in zip(parameters, anchors))
        loss = hard + 0.00001 * anchor
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % 25 == 0:
            with torch.no_grad():
                full = joint.infer(inputs, *parameters)
                wrong = int(torch.count_nonzero((full > 0) != targets))
                min_signed = float((signs * full).min())
            if wrong < best_wrong:
                best_wrong = wrong
                best_state = tuple(item.detach().numpy().copy() for item in parameters)
            if step == 1 or step % 100 == 0 or wrong < best_wrong:
                print(
                    json.dumps(
                        {
                            "step": step,
                            "loss": float(loss.detach()),
                            "wrong": wrong,
                            "best_wrong": best_wrong,
                            "min_signed": min_signed,
                        }
                    ),
                    flush=True,
                )
            if wrong == 0:
                break

    if best_state is None:
        raise RuntimeError("refinement produced no state")
    for parameter, value in zip(parameters, best_state):
        parameter.data.copy_(torch.from_numpy(value))
    save_candidate(parameters)
    result = build_blend.validate_and_score((344, "rank2_hard", str(OUT)))
    result["training_wrong"] = best_wrong
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
