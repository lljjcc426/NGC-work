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
OUT = ROOT / "reconstruction_candidates" / "b_task344_rank2_joint_v3" / "task344.onnx"


def load_examples() -> tuple[torch.Tensor, torch.Tensor]:
    data = json.loads((ROOT / "data" / "competition" / "task344.json").read_text())
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for split in ("train", "test", "arc-gen"):
        for example in data.get(split, []):
            pair = build_blend.convert_to_numpy(example)
            if pair is not None:
                inputs.append(pair["input"][0, :, :10, :10])
                targets.append(pair["output"][0, :, :10, :10])
    return torch.from_numpy(np.stack(inputs)), torch.from_numpy(np.stack(targets)).bool()


def infer(
    inputs: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    u_color: torch.Tensor,
    g_color: torch.Tensor,
    a_color: torch.Tensor,
) -> torch.Tensor:
    matrix = torch.einsum("pvt,tl->pvl", left, right)
    color = torch.einsum("ok,nk,kl->onl", u_color, g_color, a_color)
    first = torch.einsum("pvl,bnvw->bnplw", matrix, inputs)
    spatial = torch.einsum("bnplw,qwl->bnpql", first, matrix)
    return torch.einsum("bnpql,onl->bopq", spatial, color)


def save_candidate(
    left: np.ndarray,
    right: np.ndarray,
    u_color: np.ndarray,
    g_color: np.ndarray,
    a_color: np.ndarray,
) -> None:
    model = onnx.load(BASE)
    replacements = {
        "M_left": left.astype(np.float32),
        "M_right": right.astype(np.float32),
        "U": u_color.astype(np.float32),
        "G": g_color.astype(np.float32),
        "A": a_color.astype(np.float32),
    }
    kept = [item for item in model.graph.initializer if item.name not in {"M", "U", "G", "A"}]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.initializer.extend(
        numpy_helper.from_array(value, name) for name, value in replacements.items()
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
    torch.manual_seed(3443)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    model = onnx.load(BASE)
    arrays = {item.name: numpy_helper.to_array(item).astype(np.float32) for item in model.graph.initializer}
    original_m = arrays["M"]
    svd_u, singular, svd_vt = np.linalg.svd(original_m.reshape(-1, 3), full_matrices=False)
    root = np.sqrt(singular[:2])

    left = torch.nn.Parameter(
        torch.from_numpy((svd_u[:, :2] * root).reshape(10, 10, 2).astype(np.float32))
    )
    right = torch.nn.Parameter(torch.from_numpy((root[:, None] * svd_vt[:2]).astype(np.float32)))
    u_color = torch.nn.Parameter(torch.from_numpy(arrays["U"].copy()))
    g_color = torch.nn.Parameter(torch.from_numpy(arrays["G"].copy()))
    a_color = torch.nn.Parameter(torch.from_numpy(arrays["A"].copy()))

    inputs, targets = load_examples()
    target_sign = targets.float().mul(2).sub(1)
    weights = torch.where(targets, 9.0, 1.0)
    parameters = [left, right, u_color, g_color, a_color]
    optimizer = torch.optim.Adam(parameters, lr=0.003)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12000, eta_min=0.00005)

    best_wrong = targets.numel()
    best_state: tuple[np.ndarray, ...] | None = None
    batch_size = 96
    for step in range(1, 12001):
        index = torch.randint(0, inputs.shape[0], (batch_size,))
        output = infer(inputs[index], left, right, u_color, g_color, a_color)
        signed = target_sign[index] * output
        loss = (weights[index] * F.softplus(0.5 - signed)).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 10.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % 100 == 0:
            with torch.no_grad():
                full = infer(inputs, left, right, u_color, g_color, a_color)
                wrong = int(torch.count_nonzero((full > 0) != targets))
                min_positive = float(full[targets].min())
                max_negative = float(full[~targets].max())
            if wrong < best_wrong:
                best_wrong = wrong
                best_state = tuple(item.detach().numpy().copy() for item in parameters)
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": float(loss.detach()),
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

    if best_state is None:
        raise RuntimeError("training produced no state")
    save_candidate(*best_state)
    result = build_blend.validate_and_score((344, "rank2_joint", str(OUT)))
    result["training_wrong"] = best_wrong
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
