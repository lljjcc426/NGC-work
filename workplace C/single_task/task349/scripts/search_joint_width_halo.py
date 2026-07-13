from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from onnx import numpy_helper


TASK_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TASK_DIR.parents[2]
TASK_JSON = REPO_ROOT / "neurogolf_400_tasks" / "tasks" / "task349.json"
BASELINE = Path(
    r"E:/kagglegolf/submissions/candidates/"
    r"GOLF_20260711_096_v95_plus_4_compact/onnx/task349.onnx"
)
UTILS = Path(r"E:/kagglegolf/data/raw/neurogolf-2026/neurogolf_utils/neurogolf_utils.py")


def load_utils():
    spec = importlib.util.spec_from_file_location("task349_utils", UTILS)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {UTILS}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._NEUROGOLF_DIR = "E:/kagglegolf/data/raw/neurogolf-2026/"
    return module


def load_examples() -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    utils = load_utils()
    payload = json.loads(TASK_JSON.read_text(encoding="utf-8"))
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    expected: list[np.ndarray] = []
    for split in ("train", "test", "arc-gen"):
        for example in payload[split]:
            arrays = utils.convert_to_numpy(example)
            grid = arrays["input"]
            answer = arrays["output"]
            inputs.append((grid[:, 9:10] > 0).astype(np.float32))
            # The final Max restores color 9 directly. Requiring the halo branch
            # to also cover those cells adds false constraints and can make a
            # valid lower-rank factorization appear impossible.
            targets.append((answer[:, 3:4] > 0).astype(np.float32))
            expected.append(answer)
    return np.concatenate(inputs), np.concatenate(targets), expected


def ste_round(value: torch.Tensor) -> torch.Tensor:
    return value + (torch.round(value) - value).detach()


class JointWidthHalo(torch.nn.Module):
    def __init__(self, rank: int, seed: int):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.detector = torch.nn.Parameter(torch.randn(rank, 1, 1, 11, generator=generator) * 2)
        self.bias = torch.nn.Parameter(torch.zeros(rank))
        self.halo = torch.nn.Parameter(torch.randn(1, rank, 11, 20, generator=generator) * 0.2)

    def quantized(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        detector = ste_round(self.detector.clamp(-100, 100))
        bias = ste_round(self.bias.clamp(-500, 500))
        halo = ste_round(self.halo.clamp(-32, 32))
        return detector, bias, halo

    def forward(self, ch9: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        detector, bias, halo = self.quantized()
        encoded = F.conv2d(F.pad(ch9, (1, 9, 0, 0)), detector, bias)
        encoded = ste_round(encoded.clamp(0, 255))
        raw_halo = F.conv2d(F.pad(encoded, (14, 5, 5, 5)), halo)
        return raw_halo, encoded


def exact_errors(
    ch9: torch.Tensor, target: torch.Tensor, detector: np.ndarray, bias: np.ndarray, halo: np.ndarray
) -> tuple[int, int]:
    with torch.no_grad():
        encoded = F.conv2d(
            F.pad(ch9, (1, 9, 0, 0)),
            torch.from_numpy(detector).to(ch9.device, torch.float32),
            torch.from_numpy(bias).to(ch9.device, torch.float32),
        ).round().clamp(0, 255)
        score = F.conv2d(
            F.pad(encoded, (14, 5, 5, 5)),
            torch.from_numpy(halo).to(ch9.device, torch.float32),
        ).round().clamp(0, 255)
        pred = score > 0
        mismatch = pred != (target > 0)
        return int(mismatch.sum().item()), int((mismatch.flatten(1).sum(1) > 0).sum().item())


def export_candidate(detector: np.ndarray, bias: np.ndarray, halo: np.ndarray, output_path: Path) -> None:
    model = onnx.load(str(BASELINE))
    horizontal = next(node for node in model.graph.node if node.name == "h_conv")
    halo_node = next(node for node in model.graph.node if node.name == "halo_conv")
    horizontal.attribute[:] = [
        onnx.helper.make_attribute("kernel_shape", [1, 11]),
        onnx.helper.make_attribute("pads", [0, 1, 0, 9]),
        onnx.helper.make_attribute("strides", [1, 1]),
    ]
    halo_node.attribute[:] = [
        onnx.helper.make_attribute("kernel_shape", [11, 20]),
        onnx.helper.make_attribute("pads", [5, 14, 5, 5]),
        onnx.helper.make_attribute("strides", [1, 1]),
    ]
    replacements = {
        "h_kernel_combined_i8": detector.astype(np.int8),
        "h_bias_combined_i32": bias.astype(np.int32),
        "halo_weight_i8": halo.astype(np.int8),
    }
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name in replacements:
            model.graph.initializer[index].CopyFrom(
                numpy_helper.from_array(replacements[initializer.name], name=initializer.name)
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(output_path))


def train(rank: int, seed: int, steps: int, device: str) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    x_np, y_np, _ = load_examples()
    ch9 = torch.from_numpy(x_np).to(device)
    target = torch.from_numpy(y_np).to(device)
    model = JointWidthHalo(rank, seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.04)
    best = (10**9, 10**9, None, None, None)
    for step in range(steps):
        raw, encoded = model(ch9)
        positive = target > 0
        # Exact inference is positive iff the uint8 halo output is nonzero.
        loss_pos = F.relu(1.0 - raw[positive]).square().mean()
        loss_neg = F.relu(raw[~positive]).square().mean()
        loss = loss_pos + 3.0 * loss_neg + 1e-6 * encoded.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 25 == 0 or step + 1 == steps:
            detector_t, bias_t, halo_t = model.quantized()
            detector = detector_t.detach().cpu().numpy().astype(np.int8)
            bias = bias_t.detach().cpu().numpy().astype(np.int32)
            halo = halo_t.detach().cpu().numpy().astype(np.int8)
            errors, failed = exact_errors(ch9, target, detector, bias, halo)
            if (errors, failed) < best[:2]:
                best = (errors, failed, detector.copy(), bias.copy(), halo.copy())
                print(f"rank={rank} seed={seed} step={step} errors={errors} failed_examples={failed}", flush=True)
            if errors == 0:
                break
    return best  # type: ignore[return-value]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=TASK_DIR / "onnx" / "task349_joint_rank_candidate.onnx")
    args = parser.parse_args()
    overall = (10**9, 10**9, None, None, None)
    for seed in range(args.seeds):
        result = train(args.rank, seed, args.steps, args.device)
        if result[:2] < overall[:2]:
            overall = result
        if overall[0] == 0:
            break
    errors, failed, detector, bias, halo = overall
    np.savez(
        TASK_DIR / f"debug_joint_rank{args.rank}_best.npz",
        detector=detector,
        bias=bias,
        halo=halo,
        errors=errors,
        failed_examples=failed,
    )
    print(f"best rank={args.rank}: errors={errors}, failed_examples={failed}")
    if errors == 0:
        export_candidate(detector, bias, halo, args.output)
        print(args.output)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
