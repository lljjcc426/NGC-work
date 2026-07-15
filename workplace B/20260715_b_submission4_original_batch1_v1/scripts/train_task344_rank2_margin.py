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


BASE = ROOT / "team_baselines" / "team_submission4_20260715" / "extracted" / "task344.onnx"
OUT = ROOT / "reconstruction_candidates" / "b_task344_rank2_margin_v2" / "task344.onnx"


def load_examples() -> tuple[torch.Tensor, torch.Tensor]:
    data = json.loads((ROOT / "data" / "competition" / "task344.json").read_text())
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for split in ("train", "test", "arc-gen"):
        for example in data.get(split, []):
            pair = build_blend.convert_to_numpy(example)
            if pair is None:
                continue
            inputs.append(pair["input"][0, :, :10, :10])
            targets.append(pair["output"][0, :, :10, :10])
    return torch.from_numpy(np.stack(inputs)), torch.from_numpy(np.stack(targets)).bool()


def logits(
    inputs: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    color_rule: torch.Tensor,
) -> torch.Tensor:
    matrix = torch.einsum("rvt,tl->rvl", left, right)
    first = torch.einsum("rvl,bnvw->bnrlw", matrix, inputs)
    spatial = torch.einsum("bnrlw,swl->bnrsl", first, matrix)
    return torch.einsum("bnrsl,onl->bors", spatial, color_rule)


def save_candidate(left: np.ndarray, right: np.ndarray) -> None:
    model = onnx.load(BASE)
    kept = [item for item in model.graph.initializer if item.name != "M"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(left.astype(np.float32), "M_left"),
            numpy_helper.from_array(right.astype(np.float32), "M_right"),
        ]
    )
    node = model.graph.node[0]
    del node.input[:]
    node.input.extend(
        [
            "input", "E", "E", "M_left", "M_right", "M_left", "M_right",
            "E", "E", "U", "G", "A",
        ]
    )
    equation = next(attr for attr in node.attribute if attr.name == "equation")
    equation.s = b"bnij,iv,jw,pvt,tl,qwu,ul,rp,sq,ok,nk,kl->bors"
    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, OUT)


def main() -> None:
    torch.manual_seed(344)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    model = onnx.load(BASE)
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    original = arrays["M"].astype(np.float32)
    color_rule = np.einsum("ok,nk,kl->onl", arrays["U"], arrays["G"], arrays["A"])

    u, singular, vt = np.linalg.svd(original.reshape(-1, 3), full_matrices=False)
    root = np.sqrt(singular[:2])
    left = torch.nn.Parameter(
        torch.from_numpy((u[:, :2] * root).reshape(10, 10, 2).astype(np.float32))
    )
    right = torch.nn.Parameter(torch.from_numpy((root[:, None] * vt[:2]).astype(np.float32)))
    inputs, targets = load_examples()
    target_sign = targets.float().mul(2).sub(1)
    weights = torch.where(targets, 9.0, 1.0)
    color_tensor = torch.from_numpy(color_rule.astype(np.float32))
    original_tensor = torch.from_numpy(original)
    optimizer = torch.optim.Adam([left, right], lr=0.01)

    best_wrong = targets.numel()
    best_factors: tuple[np.ndarray, np.ndarray] | None = None
    batch_size = 64
    for step in range(1, 5001):
        index = torch.randint(0, inputs.shape[0], (batch_size,))
        output = logits(inputs[index], left, right, color_tensor)
        signed = target_sign[index] * output
        margin_loss = (weights[index] * F.softplus(0.25 - signed)).mean()
        reconstructed = torch.einsum("rvt,tl->rvl", left, right)
        loss = margin_loss + 0.0002 * F.mse_loss(reconstructed, original_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % 100 == 0:
            with torch.no_grad():
                full = logits(inputs, left, right, color_tensor)
                wrong = int(torch.count_nonzero((full > 0) != targets))
                min_positive = float(full[targets].min())
                max_negative = float(full[~targets].max())
            if wrong < best_wrong:
                best_wrong = wrong
                best_factors = (left.detach().numpy().copy(), right.detach().numpy().copy())
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": float(loss),
                        "wrong": wrong,
                        "best_wrong": best_wrong,
                        "min_positive": min_positive,
                        "max_negative": max_negative,
                    }
                ),
                flush=True,
            )
            if wrong == 0:
                break

    if best_factors is None:
        raise RuntimeError("training produced no factors")
    save_candidate(*best_factors)
    result = build_blend.validate_and_score((344, "rank2_margin", str(OUT)))
    result["training_wrong"] = best_wrong
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
