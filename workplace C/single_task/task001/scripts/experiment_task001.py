from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


REPO_ROOT = Path(__file__).resolve().parents[4]
TASK_ROOT = Path(__file__).resolve().parents[1]
DEBUG_DIR = TASK_ROOT / "debug"
REPORT_DIR = TASK_ROOT / "reports"
PARENT = (
    REPO_ROOT
    / "workplace C"
    / "artifacts"
    / "full400_round36_public_source_safe37"
    / "onnx"
    / "task001.onnx"
)
COMMON_DIR = REPO_ROOT / "workplace C" / "neurogolf-2026-work" / "scripts"
sys.path.insert(0, str(COMMON_DIR))

from c_score_common import load_official_utils, score_onnx  # noqa: E402


INPUT_INFO = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
OUTPUT_INFO = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])


def save_model(
    name: str,
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    *,
    opset: int = 13,
    value_info: list[onnx.ValueInfoProto] | None = None,
) -> Path:
    graph = helper.make_graph(
        nodes,
        name,
        [copy.deepcopy(INPUT_INFO)],
        [copy.deepcopy(OUTPUT_INFO)],
        initializers,
        value_info=value_info or [],
    )
    model = helper.make_model(
        graph,
        producer_name="ngc-task001-independent-experiments",
        opset_imports=[helper.make_opsetid("", opset)],
        ir_version=10,
    )
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    path = DEBUG_DIR / f"{name}.onnx"
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    return path


def build_dilated_dynamic_convtranspose(*, background_bias: float | None = None) -> Path:
    weights = np.zeros((1, 10, 2, 2), dtype=np.float32)
    weights[0, 1:, 0, 0] = 1.0
    initializers = [numpy_helper.from_array(weights, "foreground_weights")]
    ct_inputs = ["foreground_mask", "input"]
    suffix = "cost76"
    if background_bias is not None:
        bias = np.zeros(10, dtype=np.float32)
        bias[0] = background_bias
        initializers.append(numpy_helper.from_array(bias, "output_bias"))
        ct_inputs.append("output_bias")
        suffix = f"bias_{str(background_bias).replace('-', 'm').replace('.', 'p')}"
    nodes = [
        helper.make_node(
            "Conv",
            ["input", "foreground_weights"],
            ["foreground_mask"],
            kernel_shape=[2, 2],
            dilations=[27, 27],
            pads=[0, 0, 0, 0],
        ),
        helper.make_node(
            "ConvTranspose",
            ct_inputs,
            ["output"],
            strides=[3, 3],
            pads=[0, 0, 6, 6],
        ),
    ]
    return save_model(
        f"task001_dilated_dynamic_convtranspose_{suffix}",
        nodes,
        initializers,
        value_info=[
            helper.make_tensor_value_info(
                "foreground_mask", TensorProto.FLOAT, [1, 1, 3, 3]
            )
        ],
    )


def build_exact_two_channel_dynamic(*, direct_output_shape: bool = False) -> Path:
    # Macro-background stamps an all-background 3x3 kernel. Macro-foreground
    # stamps the input one-hot sprite. Their disjoint sum is the exact raw output.
    background_kernel = np.zeros((1, 10, 3, 3), dtype=np.float32)
    background_kernel[0, 0] = 1.0
    nodes = [
        helper.make_node(
            "Slice",
            ["input"],
            ["micro"],
            starts=[0, 0],
            ends=[3, 3],
            axes=[2, 3],
        ),
        helper.make_node(
            "ReduceSum", ["micro"], ["region"], axes=[1], keepdims=1
        ),
        helper.make_node("Slice", ["micro"], ["macro_background"], starts=[0], ends=[1], axes=[1]),
        helper.make_node("Sub", ["region", "macro_background"], ["macro_foreground"]),
        helper.make_node("Concat", ["macro_background", "macro_foreground"], ["macro2"], axis=1),
        helper.make_node("Concat", ["background_kernel", "micro"], ["kernel2"], axis=0),
    ]
    if direct_output_shape:
        nodes.append(
            helper.make_node(
                "ConvTranspose",
                ["macro2", "kernel2"],
                ["output"],
                strides=[3, 3],
                output_shape=[30, 30],
            )
        )
        name = "task001_exact_two_channel_dynamic_output_shape30"
    else:
        nodes.extend(
            [
                helper.make_node(
                    "ConvTranspose",
                    ["macro2", "kernel2"],
                    ["out9"],
                    strides=[3, 3],
                ),
                helper.make_node(
                    "Pad",
                    ["out9"],
                    ["output"],
                    pads=[0, 0, 0, 0, 0, 0, 21, 21],
                ),
            ]
        )
        name = "task001_exact_two_channel_dynamic"
    return save_model(
        name,
        nodes,
        [numpy_helper.from_array(background_kernel, "background_kernel")],
        opset=9,
    )


def build_rank1(alpha: float) -> Path:
    model = onnx.load(PARENT)
    dense = next(x for x in model.graph.initializer if x.name == "m")
    values = numpy_helper.to_array(dense)
    packed = (values[0] + alpha * values[1])[None].astype(np.float32)
    dense.CopyFrom(numpy_helper.from_array(packed, "m"))
    model.graph.name = f"task001_rank1_alpha_{alpha:g}"
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    tag = str(alpha).replace("-", "m").replace(".", "p")
    path = DEBUG_DIR / f"task001_rank1_alpha_{tag}.onnx"
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    return path


def session_for(path: Path) -> ort.InferenceSession:
    model = onnx.load(path)
    sanitized = load_official_utils().sanitize_model(model)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def bundled_channel_diagnostics(path: Path) -> dict[str, object]:
    payload = json.loads(
        (REPO_ROOT / "neurogolf_400_tasks" / "tasks" / "task001.json").read_text(
            encoding="utf-8"
        )
    )
    utils = load_official_utils()
    session = session_for(path)
    per_channel = np.zeros(10, dtype=np.int64)
    examples = 0
    foreground_exact = 0
    for split in ("train", "test", "arc-gen"):
        for example in payload[split]:
            arrays = utils.convert_to_numpy(example)
            actual = utils.run_network(session, arrays["input"])
            expected = arrays["output"]
            per_channel += np.count_nonzero(actual != expected, axis=(0, 2, 3))
            foreground_exact += int(np.array_equal(actual[:, 1:], expected[:, 1:]))
            examples += 1
    return {
        "examples": examples,
        "foreground_channels_exact_examples": foreground_exact,
        "mismatched_cells_by_channel": per_channel.tolist(),
    }


def exhaustive_check(path: Path) -> dict[str, int]:
    session = session_for(path)
    checked = 0
    passed = 0
    generator_checked = 0
    generator_passed = 0
    for bits in range(1 << 9):
        mask = np.asarray([(bits >> i) & 1 for i in range(9)], dtype=np.float32).reshape(3, 3)
        foreground = np.kron(mask, mask)
        for color in range(1, 10):
            model_input = np.zeros((1, 10, 30, 30), dtype=np.float32)
            model_input[0, 0, :3, :3] = 1.0 - mask
            model_input[0, color, :3, :3] = mask
            expected = np.zeros((1, 10, 30, 30), dtype=np.float32)
            expected[0, 0, :9, :9] = 1.0 - foreground
            expected[0, color, :9, :9] = foreground
            actual = (session.run(["output"], {"input": model_input})[0] > 0).astype(np.float32)
            ok = np.array_equal(actual, expected)
            checked += 1
            passed += int(ok)
            if 2 <= int(mask.sum()) <= 8:
                generator_checked += 1
                generator_passed += int(ok)
    return {
        "all_states_checked": checked,
        "all_states_passed": passed,
        "generator_states_checked": generator_checked,
        "generator_states_passed": generator_passed,
    }


def score(path: Path) -> dict[str, object]:
    result = score_onnx("task001", path, validate_all=True)
    return vars(result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exhaustive", action="store_true")
    args = parser.parse_args()
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {
        "parent": PARENT,
        "cost76": build_dilated_dynamic_convtranspose(),
        "bias_positive": build_dilated_dynamic_convtranspose(background_bias=0.5),
        "bias_negative": build_dilated_dynamic_convtranspose(background_bias=-0.5),
        "two_channel_exact": build_exact_two_channel_dynamic(),
        "two_channel_output_shape30": build_exact_two_channel_dynamic(direct_output_shape=True),
    }
    rank1_rows = []
    for alpha in (-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0):
        path = build_rank1(alpha)
        row = score(path)
        row["alpha"] = alpha
        rank1_rows.append(row)
    best_rank1 = max(rank1_rows, key=lambda row: int(row["examples_passed"]))
    paths["rank1_best"] = Path(str(best_rank1["path"]))

    report: dict[str, object] = {
        "parent_path": str(PARENT),
        "models": {},
        "rank1_sweep": rank1_rows,
        "bias_proof": {
            "raw_background_formula": "macro_fg * micro_bg + bias",
            "macro_bg_micro_fg_requires": "bias > 0",
            "macro_fg_micro_fg_requires": "bias <= 0",
            "conclusion": "No scalar ConvTranspose background bias can satisfy both states.",
        },
    }
    for label, path in paths.items():
        entry = score(path)
        entry["bundled_channel_diagnostics"] = bundled_channel_diagnostics(path)
        if args.exhaustive and label in {"parent", "cost76", "two_channel_exact", "rank1_best"}:
            entry["exhaustive"] = exhaustive_check(path)
        report["models"][label] = entry

    report_path = REPORT_DIR / "task001_experiment_results.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
