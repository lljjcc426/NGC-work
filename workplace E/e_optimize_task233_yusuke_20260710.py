#!/usr/bin/env python
"""Task233 focused optimization attempts against the yusuke 7267.31 base."""
from __future__ import annotations

import copy
import csv
import math
import pathlib
import sys
import tempfile
import zipfile

import numpy as np
import onnx
import onnxoptimizer
import onnxruntime

try:
    import onnxsim
except Exception:  # pragma: no cover
    onnxsim = None


NGC_ROOT = pathlib.Path(r"F:\kaggle\neurogolf-2026")
WORKPLACE = pathlib.Path(__file__).resolve().parent
BASE_ZIP = (
    NGC_ROOT
    / "external"
    / "source_review_20260709_e"
    / "yusuketogashi_baseline_7267_31"
    / "output"
    / "submission.zip"
)
OUT_DIR = WORKPLACE / "optimized_onnx" / "task233_20260710"
OUT_CSV = WORKPLACE / "e_task233_yusuke_opt_scan_20260710.csv"

sys.path.insert(0, str(NGC_ROOT / "data" / "neurogolf_utils"))
import neurogolf_utils as ng  # noqa: E402


ng._NEUROGOLF_DIR = str((NGC_ROOT / "data").resolve()) + "\\"
onnxruntime.set_default_logger_severity(3)


OPT_PASSES = [
    "eliminate_identity",
    "eliminate_nop_cast",
    "eliminate_deadend",
    "eliminate_unused_initializer",
    "eliminate_duplicate_initializer",
    "eliminate_common_subexpression",
    "eliminate_nop_reshape",
    "eliminate_nop_transpose",
    "eliminate_nop_pad",
    "eliminate_nop_concat",
    "eliminate_nop_flatten",
    "eliminate_nop_expand",
    "eliminate_nop_split",
    "fuse_consecutive_squeezes",
    "fuse_consecutive_unsqueezes",
    "fuse_consecutive_transposes",
    "fuse_consecutive_concats",
    "fuse_consecutive_reduce_unsqueeze",
    "fuse_consecutive_slices",
    "eliminate_shape_gather",
    "eliminate_shape_op",
    "eliminate_slice_after_shape",
    "eliminate_consecutive_idempotent_ops",
    "extract_constant_to_initializer",
]


def fix_names(model: onnx.ModelProto) -> None:
    seen: set[str] = set()
    for node in model.graph.node:
        base = node.name or (node.output[0] if node.output else "node")
        name = base
        suffix = 0
        while name in seen:
            suffix += 1
            name = f"{base}_{suffix}"
        node.name = name
        seen.add(name)


def prune(model: onnx.ModelProto) -> None:
    needed = {output.name for output in model.graph.output}
    kept_reversed = []
    for node in reversed(model.graph.node):
        if any(output in needed for output in node.output):
            kept_reversed.append(node)
            needed.update(value for value in node.input if value)
    kept_outputs = {tuple(node.output) for node in kept_reversed}
    kept = [node for node in model.graph.node if tuple(node.output) in kept_outputs]
    used = {value for node in kept for value in node.input if value}
    kept_initializers = [init for init in model.graph.initializer if init.name in used]
    del model.graph.node[:]
    model.graph.node.extend(kept)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)


def score_cost(model: onnx.ModelProto) -> int | None:
    sanitized = ng.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return None
    fix_names(sanitized)
    with tempfile.TemporaryDirectory() as tmp:
        options = onnxruntime.SessionOptions()
        options.enable_profiling = True
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.log_severity_level = 3
        options.profile_file_prefix = str(pathlib.Path(tmp) / "task233")
        try:
            session = onnxruntime.InferenceSession(sanitized.SerializeToString(), options)
        except Exception:
            return None
        examples = ng.load_examples(233)
        batch = ng.convert_to_numpy(examples["train"][0])
        if batch is not None:
            try:
                ng.run_network(session, batch["input"])
            except Exception:
                pass
        trace_path = session.end_profiling()
        try:
            memory, params = ng.score_network(sanitized, trace_path)
        except Exception:
            return None
    if memory is None or params is None or memory < 0 or params < 0:
        return None
    return int(memory) + int(params)


def verify_all(model: onnx.ModelProto) -> bool:
    sanitized = ng.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return False
    fix_names(sanitized)
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    try:
        session = onnxruntime.InferenceSession(sanitized.SerializeToString(), options)
    except Exception:
        return False
    examples = ng.load_examples(233)
    for example in examples["train"] + examples["test"] + examples["arc-gen"]:
        batch = ng.convert_to_numpy(example)
        if batch is None:
            continue
        try:
            actual = ng.run_network(session, batch["input"])
        except Exception:
            return False
        if not np.array_equal(actual, batch["output"]):
            return False
    return True


def sample_ok(model: onnx.ModelProto, arc_gen_limit: int = 30) -> bool:
    sanitized = ng.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return False
    fix_names(sanitized)
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    try:
        session = onnxruntime.InferenceSession(sanitized.SerializeToString(), options)
    except Exception:
        return False
    examples = ng.load_examples(233)
    for example in examples["train"] + examples["test"] + examples["arc-gen"][:arc_gen_limit]:
        batch = ng.convert_to_numpy(example)
        if batch is None:
            continue
        try:
            actual = ng.run_network(session, batch["input"])
        except Exception:
            return False
        if not np.array_equal(actual, batch["output"]):
            return False
    return True


def candidate_models(base: onnx.ModelProto):
    yield "base", base
    try:
        yield "onnxoptimizer", onnxoptimizer.optimize(base, OPT_PASSES)
    except Exception:
        pass
    if onnxsim is not None:
        try:
            simplified, ok = onnxsim.simplify(
                base, overwrite_input_shapes=[("input", [1, 10, 30, 30])]
            )
            if ok:
                yield "onnxsim", simplified
        except Exception:
            pass

    concat_pairs = []
    for idx, node in enumerate(base.graph.node):
        if node.op_type == "Concat" and len(node.input) == 9:
            concat_pairs.append(idx)
    for count in range(5, 9):
        model = copy.deepcopy(base)
        for idx in concat_pairs:
            node = model.graph.node[idx]
            del node.input[count:]
        prune(model)
        yield f"patch_first{count}", model
    for drop in range(9):
        model = copy.deepcopy(base)
        for idx in concat_pairs:
            node = model.graph.node[idx]
            kept = [value for i, value in enumerate(node.input) if i != drop]
            del node.input[:]
            node.input.extend(kept)
        prune(model)
        yield f"patch_drop{drop + 1}", model

    for idx, node in enumerate(base.graph.node):
        if node.op_type not in {"Where", "And"}:
            continue
        choices = range(1, 3) if node.op_type == "Where" else range(2)
        for input_index in choices:
            if input_index >= len(node.input):
                continue
            model = copy.deepcopy(base)
            old = model.graph.node[idx].output[0]
            replacement = model.graph.node[idx].input[input_index]
            if not replacement:
                continue
            for other in model.graph.node:
                for pos, value in enumerate(other.input):
                    if value == old:
                        other.input[pos] = replacement
            prune(model)
            yield f"bypass_n{idx:03d}_{node.op_type}_i{input_index}", model


def points(cost: int) -> float:
    return max(1.0, 25.0 - math.log(cost))


def main() -> int:
    with zipfile.ZipFile(BASE_ZIP) as zf:
        base = onnx.load_from_string(zf.read("task233.onnx"))
    base_cost = score_cost(base)
    if base_cost is None:
        raise RuntimeError("could not score base")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    seen: set[bytes] = set()
    for name, model in candidate_models(base):
        payload = model.SerializeToString()
        if payload in seen:
            continue
        seen.add(payload)
        status = "not_checked"
        cost = score_cost(model)
        full_ok = False
        if cost is None:
            status = "score_error"
        elif cost >= base_cost:
            status = "not_better"
        elif not sample_ok(model):
            status = "sample_wrong"
        else:
            full_ok = verify_all(model)
            status = "ok_improved" if full_ok else "full_wrong"
            if full_ok:
                onnx.save(model, OUT_DIR / f"{name}.onnx")
        rows.append(
            {
                "candidate": name,
                "base_cost": base_cost,
                "candidate_cost": "" if cost is None else cost,
                "delta_cost": "" if cost is None else base_cost - cost,
                "base_points": f"{points(base_cost):.9f}",
                "candidate_points": "" if cost is None else f"{points(cost):.9f}",
                "delta_points": "" if cost is None else f"{points(cost) - points(base_cost):.9f}",
                "status": status,
                "verified_all": full_ok,
                "nodes": len(model.graph.node),
                "initializers": len(model.graph.initializer),
            }
        )
        print(f"{name}: status={status} cost={cost}", flush=True)

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {OUT_CSV}")
    print(f"Improved={sum(row['status'] == 'ok_improved' for row in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
