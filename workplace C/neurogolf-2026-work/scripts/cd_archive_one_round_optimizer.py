from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve()
PROJECT = HERE.parents[1]
WORKPLACE_C = PROJECT.parent
DEFAULT_ARCHIVE = PROJECT / "data" / "external" / "neurogolf7300_archive" / "onnx"
DEFAULT_CANDIDATES = WORKPLACE_C / "artifacts" / "cd_archive_one_round"
DEFAULT_BENCHMARK = WORKPLACE_C / "score_docs" / "41_CD_NEUROGOLF7300_BENCHMARK.csv"
DEFAULT_OUTPUT = WORKPLACE_C / "score_docs" / "42_CD_ARCHIVE_ONE_ROUND_RESULTS.csv"


def _attribute(node: onnx.NodeProto, name: str) -> onnx.AttributeProto | None:
    return next((item for item in node.attribute if item.name == name), None)


def _initializer_key(item: onnx.TensorProto) -> str:
    array = numpy_helper.to_array(item)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def deduplicate_initializers(model: onnx.ModelProto) -> int:
    canonical: dict[str, str] = {}
    replacements: dict[str, str] = {}
    kept: list[onnx.TensorProto] = []
    for item in model.graph.initializer:
        key = _initializer_key(item)
        if key in canonical:
            replacements[item.name] = canonical[key]
        else:
            canonical[key] = item.name
            kept.append(item)
    if not replacements:
        return 0
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name in replacements:
                node.input[index] = replacements[name]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return len(replacements)


def prune_initializers(model: onnx.ModelProto) -> int:
    used = {name for node in model.graph.node for name in node.input if name}
    kept = [item for item in model.graph.initializer if item.name in used]
    removed = len(model.graph.initializer) - len(kept)
    if removed:
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept)
    return removed


def _zero_border(weight: np.ndarray) -> tuple[int, int, int, int]:
    if weight.ndim != 4:
        return 0, 0, 0, 0
    support = np.any(weight != 0, axis=(0, 1))
    if not support.any():
        return 0, 0, 0, 0
    rows = np.flatnonzero(support.any(axis=1))
    cols = np.flatnonzero(support.any(axis=0))
    return int(rows[0]), int(weight.shape[2] - rows[-1] - 1), int(cols[0]), int(weight.shape[3] - cols[-1] - 1)


def crop_conv_support(model: onnx.ModelProto) -> int:
    initializers = {item.name: item for item in model.graph.initializer}
    consumer_count: dict[str, int] = {}
    for graph_node in model.graph.node:
        for name in graph_node.input:
            if name:
                consumer_count[name] = consumer_count.get(name, 0) + 1
    cropped = 0
    for node in model.graph.node:
        if node.op_type not in {"Conv", "QLinearConv"}:
            continue
        auto_pad = _attribute(node, "auto_pad")
        if auto_pad is not None and auto_pad.s not in {b"", b"NOTSET"}:
            continue
        weight_index = 1 if node.op_type == "Conv" else 3
        if len(node.input) <= weight_index or node.input[weight_index] not in initializers:
            continue
        if consumer_count.get(node.input[weight_index], 0) != 1:
            continue
        tensor = initializers[node.input[weight_index]]
        weight = numpy_helper.to_array(tensor)
        top, bottom, left, right = _zero_border(weight)
        if top + bottom + left + right == 0:
            continue
        pads = _attribute(node, "pads")
        values = list(pads.ints) if pads is not None else [0, 0, 0, 0]
        if len(values) != 4:
            continue
        dilation = _attribute(node, "dilations")
        dilation_values = list(dilation.ints) if dilation is not None else [1, 1]
        if len(dilation_values) != 2:
            continue
        trimmed = weight[:, :, top : weight.shape[2] - bottom, left : weight.shape[3] - right]
        tensor.CopyFrom(numpy_helper.from_array(trimmed, name=tensor.name))
        values = [
            values[0] - top * dilation_values[0],
            values[1] - left * dilation_values[1],
            values[2] - bottom * dilation_values[0],
            values[3] - right * dilation_values[1],
        ]
        if pads is None:
            pads = node.attribute.add()
            pads.name = "pads"
            pads.type = onnx.AttributeProto.INTS
        pads.ints[:] = values
        kernel = _attribute(node, "kernel_shape")
        if kernel is not None:
            kernel.ints[:] = [trimmed.shape[2], trimmed.shape[3]]
        cropped += 1
    return cropped


def optimize(source: Path, output: Path) -> tuple[list[str], str]:
    model = onnx.load(str(source))
    changes: list[str] = []
    count = deduplicate_initializers(model)
    if count:
        changes.append(f"dedup_init:{count}")
    count = crop_conv_support(model)
    if count:
        changes.append(f"crop_conv:{count}")
    count = prune_initializers(model)
    if count:
        changes.append(f"prune_init:{count}")
    if not changes:
        return changes, ""
    model.producer_name = "ngc_cd_archive_one_round"
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        onnx.checker.check_model(model, full_check=True)
        onnx.save(model, str(output))
        return changes, ""
    except Exception as exc:
        return changes, f"checker:{type(exc).__name__}:{exc}"


def run_task(args: tuple[dict[str, str], str, str]) -> dict[str, object]:
    benchmark, archive_dir, candidate_dir = args
    task = benchmark["task"]
    source = Path(archive_dir) / f"{task}.onnx"
    output = Path(candidate_dir) / f"{task}.onnx"
    row: dict[str, object] = {
        "task": task,
        "archive_cost": benchmark.get("archive_cost", ""),
        "parent_cost": benchmark.get("parent_cost", ""),
        "archive_valid": benchmark.get("archive_ok", ""),
        "compliance_hold": True,
        "source_path": str(source),
        "candidate_path": str(output),
    }
    if not source.exists():
        row["status"] = "missing_archive"
        return row
    changes, error = optimize(source, output)
    row["changes"] = ";".join(changes)
    if error:
        row["status"] = "checker_failed"
        row["error"] = error
        return row
    if not changes:
        row["status"] = "unchanged"
        return row

    sys.path.insert(0, str(HERE.parent))
    from c_score_common import score_onnx

    score = score_onnx(task, output, True)
    row.update({f"candidate_{key}": value for key, value in asdict(score).items()})
    try:
        archive_cost = int(float(str(benchmark.get("archive_cost", ""))))
        parent_cost = int(float(str(benchmark.get("parent_cost", ""))))
    except ValueError:
        archive_cost = parent_cost = -1
    row["delta_vs_archive"] = archive_cost - score.cost if score.cost is not None and archive_cost >= 0 else ""
    row["delta_vs_parent"] = parent_cost - score.cost if score.cost is not None and parent_cost >= 0 else ""
    row["accepted_derived"] = bool(
        score.ok
        and score.cost is not None
        and archive_cost >= 0
        and parent_cost >= 0
        and score.cost < archive_cost
        and score.cost < parent_cost
    )
    row["status"] = "accepted_derived" if row["accepted_derived"] else "rejected"
    return row


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    with args.benchmark.open(newline="", encoding="utf-8-sig") as handle:
        benchmark = list(csv.DictReader(handle))
    rows: list[dict[str, object]] = []
    jobs = [(row, str(args.archive_dir), str(args.candidate_dir)) for row in benchmark]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_task, job): job[0]["task"] for job in jobs}
        for index, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {"task": task, "status": "worker_failed", "error": f"{type(exc).__name__}:{exc}"}
            rows.append(row)
            rows.sort(key=lambda item: str(item["task"]))
            write_rows(args.output, rows)
            print(json.dumps({"completed": index, "task": task, "status": row.get("status"), "delta": row.get("delta_vs_archive", "")}), flush=True)

    summary = {
        "tasks": len(rows),
        "changed": sum(bool(row.get("changes")) for row in rows),
        "accepted_derived": sum(bool(row.get("accepted_derived")) for row in rows),
        "checker_failed": sum(row.get("status") == "checker_failed" for row in rows),
        "worker_failed": sum(row.get("status") == "worker_failed" for row in rows),
        "compliance_hold": True,
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
