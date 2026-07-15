from __future__ import annotations

import copy
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SCRIPTS = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from c_score_common import TASK_DATA_DIR, load_official_utils, score_onnx  # noqa: E402


TASK = "task086"
PARENT = (
    REPO
    / "workplace C"
    / "artifacts"
    / "full400_round36_public_source_safe37"
    / "onnx"
    / "task086.onnx"
)
NEGATIVE_PAD_SOURCE = (
    REPO
    / "workplace C"
    / "artifacts"
    / "public_source_all_candidates"
    / "task086.onnx"
)
SAFE_BASE = (
    REPO
    / "workplace C"
    / "artifacts"
    / "safe_dilated_crop_conv"
    / "task086.onnx"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def attributes(node: onnx.NodeProto) -> dict[str, object]:
    return {item.name: helper.get_attribute_value(item) for item in node.attribute}


def save_checked(model: onnx.ModelProto, path: Path) -> None:
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, path)


def remove_unused_initializers(model: onnx.ModelProto) -> list[str]:
    used = {name for node in model.graph.node for name in node.input if name}
    removed = [item.name for item in model.graph.initializer if item.name not in used]
    kept = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return removed


def fold_occupancy_to_sign(base: onnx.ModelProto) -> onnx.ModelProto:
    model = copy.deepcopy(base)
    nodes = list(model.graph.node)
    greater_index = next(
        index
        for index, node in enumerate(nodes)
        if node.op_type == "Greater" and list(node.output) == ["Fb"]
    )
    cast_index = next(
        index
        for index, node in enumerate(nodes)
        if node.op_type == "Cast" and list(node.input) == ["Fb"]
    )
    if list(nodes[cast_index].output) != ["F8"]:
        raise RuntimeError("unexpected occupancy Cast output")
    replacement = helper.make_node("Sign", ["base"], ["F8"], name="occupancy_sign")
    nodes[greater_index] = replacement
    del nodes[cast_index]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    remove_unused_initializers(model)
    model.graph.name = "task086_safe_sign_occupancy"
    return model


def encode_topk_and_pad_as_opset10(base: onnx.ModelProto) -> onnx.ModelProto:
    model = copy.deepcopy(base)
    nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        if node.op_type == "TopK":
            if list(node.input) != ["cnt", "k3"]:
                raise RuntimeError(f"unexpected TopK inputs: {list(node.input)}")
            nodes.append(
                helper.make_node(
                    "TopK",
                    ["cnt"],
                    list(node.output),
                    name=node.name or "top3_colors",
                    axis=1,
                    k=3,
                )
            )
        elif node.op_type == "Pad":
            if list(node.input) != ["label", "pad30", "pad_val"]:
                raise RuntimeError(f"unexpected Pad inputs: {list(node.input)}")
            nodes.append(
                helper.make_node(
                    "Pad",
                    ["label"],
                    list(node.output),
                    name=node.name or "pad_label_to_30",
                    mode="constant",
                    pads=[0, 0, 0, 0, 0, 0, 18, 18],
                    value=255.0,
                )
            )
        else:
            nodes.append(copy.deepcopy(node))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.opset_import[:]
    model.opset_import.extend([helper.make_opsetid("", 10)])
    remove_unused_initializers(model)
    model.graph.name = "task086_safe_sign_opset10_attrs"
    return model


def rewrite_color_permutation_with_scatter(base: onnx.ModelProto) -> onnx.ModelProto:
    model = copy.deepcopy(base)
    replaced_outputs = {
        "c1i",
        "c0i",
        "c1u",
        "c0u",
        "e1",
        "w1",
        "e0",
        "chan_perm",
    }
    nodes: list[onnx.NodeProto] = []
    inserted = False
    for node in model.graph.node:
        if node.output and node.output[0] in replaced_outputs:
            continue
        nodes.append(copy.deepcopy(node))
        if node.op_type == "TopK":
            nodes.extend(
                [
                    helper.make_node("Cast", ["ti"], ["ti_u"], name="top3_u8", to=TensorProto.UINT8),
                    helper.make_node(
                        "Gather",
                        ["ti_u", "swap_perm"],
                        ["swap_u"],
                        name="swap_nonbackground_updates",
                        axis=1,
                    ),
                    helper.make_node(
                        "Gather",
                        ["swap_u", "i1"],
                        ["c0u"],
                        name="second_color_u8",
                        axis=1,
                    ),
                    helper.make_node(
                        "ScatterElements",
                        ["chan_ids", "ti", "swap_u"],
                        ["chan_perm"],
                        name="swap_dynamic_color_ids",
                        axis=1,
                    ),
                ]
            )
            inserted = True
    if not inserted:
        raise RuntimeError("TopK insertion point not found")
    model.graph.initializer.extend(
        [numpy_helper.from_array(np.asarray([0, 2, 1], dtype=np.int64), "swap_perm")]
    )
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    remove_unused_initializers(model)
    model.graph.name = "task086_safe_scatter_color_permutation"
    return model


def compact_pad_axes(base: onnx.ModelProto) -> onnx.ModelProto:
    model = copy.deepcopy(base)
    nodes: list[onnx.NodeProto] = []
    replaced = False
    for node in model.graph.node:
        if node.op_type != "Pad":
            nodes.append(copy.deepcopy(node))
            continue
        if list(node.input) != ["label", "pad30", "pad_val"]:
            raise RuntimeError(f"unexpected Pad inputs: {list(node.input)}")
        nodes.append(
            helper.make_node(
                "Pad",
                ["label", "pad_hw", "pad_val", "axes_hw"],
                list(node.output),
                name=node.name or "pad_label_to_30",
                mode="constant",
            )
        )
        replaced = True
    if not replaced:
        raise RuntimeError("Pad node not found")
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(np.asarray([0, 0, 18, 18], dtype=np.int64), "pad_hw"),
            numpy_helper.from_array(np.asarray([2, 3], dtype=np.int64), "axes_hw"),
        ]
    )
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.opset_import[:]
    model.opset_import.extend([helper.make_opsetid("", 18)])
    remove_unused_initializers(model)
    model.graph.name = "task086_safe_compact_pad_axes"
    return model


def sparsify_initializer(model: onnx.ModelProto, name: str) -> dict[str, object]:
    dense = next((item for item in model.graph.initializer if item.name == name), None)
    if dense is None:
        raise KeyError(name)
    array = numpy_helper.to_array(dense)
    coordinates = np.argwhere(array != 0).astype(np.int64)
    values = array[tuple(coordinates.T)]
    if not len(values):
        raise RuntimeError(f"cannot create zero-nnz sparse initializer: {name}")
    sparse = helper.make_sparse_tensor(
        numpy_helper.from_array(values, name),
        numpy_helper.from_array(coordinates, f"{name}_indices"),
        list(array.shape),
    )
    kept = [item for item in model.graph.initializer if item.name != name]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.sparse_initializer.extend([sparse])
    return {
        "name": name,
        "dense_elements": int(array.size),
        "sparse_values": int(values.size),
        "saved_params": int(array.size - values.size),
    }


def stabilize_sparse_initializer_names(model: onnx.ModelProto) -> dict[str, str]:
    """Choose names that remain unchanged by the official sanitizer."""
    sparse_names = {item.values.name for item in model.graph.sparse_initializer}
    mapping: dict[str, str] = {}
    counter = 0
    for initializer in model.graph.initializer:
        if initializer.name not in mapping:
            mapping[initializer.name] = f"safe_name_{counter}"
            counter += 1

    assigned: dict[str, str] = {}
    for node in model.graph.node:
        for name in [*node.input, *node.output]:
            if not name or name in {"input", "output"}:
                continue
            if name in sparse_names:
                if name not in assigned:
                    assigned[name] = f"safe_name_{counter}"
                    counter += 1
                continue
            if name not in mapping:
                mapping[name] = f"safe_name_{counter}"
                counter += 1

    for sparse in model.graph.sparse_initializer:
        sparse.values.name = assigned[sparse.values.name]
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name in assigned:
                node.input[index] = assigned[name]
    return assigned


def examples() -> list[dict[str, object]]:
    payload = json.loads((TASK_DATA_DIR / f"{TASK}.json").read_text(encoding="utf-8"))
    return [item for split in ("train", "test", "arc-gen") for item in payload[split]]


def session(model: onnx.ModelProto, extra_outputs: list[str] | None = None) -> ort.InferenceSession:
    instrumented = copy.deepcopy(model)
    if extra_outputs:
        inferred = onnx.shape_inference.infer_shapes(instrumented, strict_mode=True)
        tensors = {
            item.name: item
            for item in [*inferred.graph.value_info, *inferred.graph.output]
        }
        existing = {item.name for item in instrumented.graph.output}
        for name in extra_outputs:
            if name not in existing:
                instrumented.graph.output.extend([copy.deepcopy(tensors[name])])
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    return ort.InferenceSession(
        instrumented.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def bundled_equivalence(left: onnx.ModelProto, right: onnx.ModelProto) -> dict[str, object]:
    utils = load_official_utils()
    left_session = session(left)
    right_session = session(right)
    checked = equal = 0
    for example in examples():
        arrays = utils.convert_to_numpy(example)
        if arrays is None:
            continue
        checked += 1
        left_value = left_session.run(["output"], {"input": arrays["input"]})[0]
        right_value = right_session.run(["output"], {"input": arrays["input"]})[0]
        equal += int(np.array_equal(left_value > 0, right_value > 0))
    return {"checked": checked, "equal": equal, "all_equal": equal == checked}


def bundled_common_tensor_equivalence(
    left: onnx.ModelProto, right: onnx.ModelProto
) -> dict[str, object]:
    left_outputs = {name for node in left.graph.node for name in node.output if name}
    right_outputs = {name for node in right.graph.node for name in node.output if name}
    common = sorted(left_outputs & right_outputs)
    left_session = session(left, common)
    right_session = session(right, common)
    utils = load_official_utils()
    checked = equal = 0
    for example in examples():
        arrays = utils.convert_to_numpy(example)
        if arrays is None:
            continue
        left_values = left_session.run(common, {"input": arrays["input"]})
        right_values = right_session.run(common, {"input": arrays["input"]})
        checked += len(common)
        equal += sum(
            int(np.array_equal(left_value, right_value))
            for left_value, right_value in zip(left_values, right_values)
        )
    return {
        "tensor_names": common,
        "comparisons": checked,
        "equal": equal,
        "all_equal": equal == checked,
    }


def structural_safe_crop_evidence() -> dict[str, object]:
    old = onnx.load(NEGATIVE_PAD_SOURCE)
    new = onnx.load(SAFE_BASE)
    old_conv = old.graph.node[0]
    new_conv = new.graph.node[0]
    old_inits = {item.name: numpy_helper.to_array(item) for item in old.graph.initializer}
    new_inits = {item.name: numpy_helper.to_array(item) for item in new.graph.initializer}
    old_weight = old_inits[old_conv.input[1]]
    new_weight = new_inits[new_conv.input[1]]
    weight_exact = bool(
        new_weight.shape == (1, 10, 2, 2)
        and np.array_equal(new_weight[:, :, 0, 0], old_weight[:, :, 0, 0])
        and np.count_nonzero(new_weight[:, :, 0, 1:]) == 0
        and np.count_nonzero(new_weight[:, :, 1, :]) == 0
    )
    geometry_exact = bool(
        attributes(old_conv).get("pads") == [0, 0, -18, -18]
        and attributes(new_conv).get("pads") == [0, 0, 0, 0]
        and attributes(new_conv).get("dilations") == [18, 18]
        and attributes(new_conv).get("kernel_shape") == [2, 2]
        and (30 - (2 - 1) * 18 - 1 + 1) == 12
    )

    common_outputs = [name for node in old.graph.node for name in node.output if name]
    old_session = session(old, common_outputs)
    new_session = session(new, common_outputs)
    utils = load_official_utils()
    intermediate_checked = 0
    intermediate_equal = 0
    for example in examples():
        arrays = utils.convert_to_numpy(example)
        if arrays is None:
            continue
        left = old_session.run(common_outputs, {"input": arrays["input"]})
        right = new_session.run(common_outputs, {"input": arrays["input"]})
        intermediate_checked += len(common_outputs)
        intermediate_equal += sum(
            int(np.array_equal(left_value, right_value))
            for left_value, right_value in zip(left, right)
        )

    rng = np.random.default_rng(860315)
    old_lin = session(old, ["Lin"])
    new_lin = session(new, ["Lin"])
    random_checked = random_close = 0
    max_abs_diff = 0.0
    for _ in range(64):
        value = rng.normal(size=(1, 10, 30, 30)).astype(np.float32)
        left = old_lin.run(["Lin"], {"input": value})[0]
        right = new_lin.run(["Lin"], {"input": value})[0]
        random_checked += 1
        difference = float(np.max(np.abs(left - right)))
        max_abs_diff = max(max_abs_diff, difference)
        random_close += int(np.allclose(left, right, rtol=1e-6, atol=1e-5))
    return {
        "source_sha256": sha256(NEGATIVE_PAD_SOURCE),
        "safe_sha256": sha256(SAFE_BASE),
        "weight_zero_extension_exact": weight_exact,
        "geometry_exact": geometry_exact,
        "bundled_intermediate_tensors_checked": intermediate_checked,
        "bundled_intermediate_tensors_equal": intermediate_equal,
        "bundled_all_intermediates_equal": intermediate_equal == intermediate_checked,
        "random_dense_inputs_checked": random_checked,
        "random_lin_allclose": random_close,
        "random_all_lin_allclose": random_close == random_checked,
        "random_lin_max_abs_diff": max_abs_diff,
    }


def model_summary(path: Path) -> dict[str, object]:
    model = onnx.load(path)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "opset": [(item.domain, item.version) for item in model.opset_import],
        "nodes": len(model.graph.node),
        "dense_initializers": len(model.graph.initializer),
        "sparse_initializers": len(model.graph.sparse_initializer),
    }


def main() -> int:
    safe = onnx.load(SAFE_BASE)
    variants: list[tuple[str, onnx.ModelProto, list[dict[str, object]]]] = []

    sign = fold_occupancy_to_sign(safe)
    variants.append(("task086_sign.onnx", sign, [{"rewrite": "Greater+Cast to Sign(base)"}]))

    opset10 = encode_topk_and_pad_as_opset10(sign)
    variants.append(
        (
            "task086_sign_opset10.onnx",
            opset10,
            [{"rewrite": "TopK K and Pad geometry/value inputs to opset10 attributes"}],
        )
    )

    scatter = rewrite_color_permutation_with_scatter(safe)
    variants.append(
        (
            "task086_scatter_colors.onnx",
            scatter,
            [
                {
                    "rewrite": "eight-node dynamic color swap to Cast+Gather+Gather+ScatterElements",
                    "predicted_memory_delta": -41,
                    "predicted_param_delta": 2,
                }
            ],
        )
    )

    pad_axes = compact_pad_axes(safe)
    variants.append(
        (
            "task086_pad_axes.onnx",
            pad_axes,
            [
                {
                    "rewrite": "Pad-18 axes input reduces full-rank pads from 8 to 4+2 elements",
                    "predicted_memory_delta": 0,
                    "predicted_param_delta": -2,
                }
            ],
        )
    )

    combined = compact_pad_axes(scatter)
    combined.graph.name = "task086_safe_scatter_colors_compact_pad"
    variants.append(
        (
            "task086_best.onnx",
            combined,
            [
                {
                    "rewrite": "combined safe color ScatterElements and compact Pad axes",
                    "predicted_cost_delta": -41,
                }
            ],
        )
    )

    sparse_w = copy.deepcopy(safe)
    sparse_w_detail = sparsify_initializer(sparse_w, "w_collapse")
    sparse_w_names = stabilize_sparse_initializer_names(sparse_w)
    sparse_w.graph.name = "task086_safe_sparse_collapse"
    variants.append(
        (
            "task086_sparse_w.onnx",
            sparse_w,
            [
                {
                    "rewrite": "dense collapse weight to exact sparse initializer",
                    **sparse_w_detail,
                    "sanitizer_stable_names": sparse_w_names,
                }
            ],
        )
    )

    sparse_both = copy.deepcopy(sparse_w)
    sparse_k_detail = sparsify_initializer(sparse_both, "karm")
    sparse_both_names = stabilize_sparse_initializer_names(sparse_both)
    sparse_both.graph.name = "task086_safe_sparse_weights"
    variants.append(
        (
            "task086_sparse_weights.onnx",
            sparse_both,
            [
                {
                    "rewrite": "dense arm kernel to exact sparse initializer",
                    **sparse_k_detail,
                    "sanitizer_stable_names": sparse_both_names,
                }
            ],
        )
    )

    sparse_all = copy.deepcopy(sparse_both)
    sparse_pad_detail = sparsify_initializer(sparse_all, "pad30")
    sparse_channel_detail = sparsify_initializer(sparse_all, "chan_ids")
    sparse_all_names = stabilize_sparse_initializer_names(sparse_all)
    sparse_all.graph.name = "task086_safe_sparse_initializers"
    variants.append(
        (
            "task086_sparse_all.onnx",
            sparse_all,
            [
                {
                    "rewrite": "dense geometry and channel constants to exact sparse initializers",
                    "parts": [sparse_pad_detail, sparse_channel_detail],
                    "sanitizer_stable_names": sparse_all_names,
                }
            ],
        )
    )

    results: list[dict[str, object]] = []
    safe_model = onnx.load(SAFE_BASE)
    for filename, model, rewrites in variants:
        path = HERE / filename
        try:
            save_checked(model, path)
            score = score_onnx(TASK, path, validate_all=True)
            equivalence = bundled_equivalence(safe_model, model) if score.ok else None
            common_equivalence = (
                bundled_common_tensor_equivalence(safe_model, model) if score.ok else None
            )
            results.append(
                {
                    "name": filename,
                    "rewrites": rewrites,
                    "model": model_summary(path),
                    "official_score": asdict(score),
                    "safe_base_output_equivalence": equivalence,
                    "safe_base_common_tensor_equivalence": common_equivalence,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "name": filename,
                    "rewrites": rewrites,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    parent_score = score_onnx(TASK, PARENT, validate_all=True)
    safe_score = score_onnx(TASK, SAFE_BASE, validate_all=True)
    report = {
        "task": TASK,
        "parent_ref": 54715535,
        "parent": {**model_summary(PARENT), "official_score": asdict(parent_score)},
        "safe_base": {**model_summary(SAFE_BASE), "official_score": asdict(safe_score)},
        "parent_to_safe_output_equivalence": bundled_equivalence(
            onnx.load(PARENT), onnx.load(SAFE_BASE)
        ),
        "safe_crop_structural_evidence": structural_safe_crop_evidence(),
        "variants": results,
    }
    output = HERE / "results.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
