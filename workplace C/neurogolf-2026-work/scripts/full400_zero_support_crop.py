from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
DEFAULT_PARENT = Path(
    r"E:/kagglegolf/submissions/candidates/"
    r"GOLF_20260713_SUBMISSION8_REBASE/onnx"
)
DEFAULT_WORK = REPO / "workplace C" / "artifacts" / "full400_zero_support_crop"
ASSIGNMENTS = REPO / "assignments" / "task_assignment_400.csv"


def _attribute(node: onnx.NodeProto, name: str) -> onnx.AttributeProto | None:
    return next((item for item in node.attribute if item.name == name), None)


def _owners() -> dict[str, str]:
    with ASSIGNMENTS.open(newline="", encoding="utf-8-sig") as handle:
        return {row["task"]: row["owner"] for row in csv.DictReader(handle)}


def _canonical(owner: str, task: str) -> Path:
    return (
        REPO
        / f"workplace {owner}"
        / "single_task"
        / task
        / "onnx"
        / f"{task}_candidate.onnx"
    )


def _zero_border(weight: np.ndarray) -> tuple[int, int, int, int]:
    if weight.ndim != 4:
        return 0, 0, 0, 0
    support = np.any(weight != 0, axis=(0, 1))
    if not support.any():
        return 0, 0, 0, 0
    rows = np.flatnonzero(support.any(axis=1))
    cols = np.flatnonzero(support.any(axis=0))
    return (
        int(rows[0]),
        int(weight.shape[2] - rows[-1] - 1),
        int(cols[0]),
        int(weight.shape[3] - cols[-1] - 1),
    )


def crop_zero_support(model: onnx.ModelProto) -> list[dict]:
    initializers = {item.name: item for item in model.graph.initializer}
    consumers: dict[str, int] = {}
    for node in model.graph.node:
        for name in node.input:
            if name:
                consumers[name] = consumers.get(name, 0) + 1

    changes: list[dict] = []
    for index, node in enumerate(model.graph.node):
        if node.op_type not in {"Conv", "QLinearConv"}:
            continue
        auto_pad = _attribute(node, "auto_pad")
        if auto_pad is not None and auto_pad.s not in {b"", b"NOTSET"}:
            continue
        weight_index = 1 if node.op_type == "Conv" else 3
        if len(node.input) <= weight_index:
            continue
        tensor = initializers.get(node.input[weight_index])
        if tensor is None or consumers.get(tensor.name, 0) != 1:
            continue
        weight = numpy_helper.to_array(tensor)
        top, bottom, left, right = _zero_border(weight)
        if top + bottom + left + right == 0:
            continue
        pads = _attribute(node, "pads")
        old_pads = list(pads.ints) if pads is not None else [0, 0, 0, 0]
        dilation = _attribute(node, "dilations")
        dilations = list(dilation.ints) if dilation is not None else [1, 1]
        if len(old_pads) != 4 or len(dilations) != 2:
            continue
        new_pads = [
            old_pads[0] - top * dilations[0],
            old_pads[1] - left * dilations[1],
            old_pads[2] - bottom * dilations[0],
            old_pads[3] - right * dilations[1],
        ]
        trimmed = weight[
            :,
            :,
            top : weight.shape[2] - bottom,
            left : weight.shape[3] - right,
        ]
        if not trimmed.size:
            continue
        tensor.CopyFrom(numpy_helper.from_array(trimmed, name=tensor.name))
        if pads is None:
            pads = node.attribute.add()
            pads.name = "pads"
            pads.type = onnx.AttributeProto.INTS
        pads.ints[:] = new_pads
        kernel = _attribute(node, "kernel_shape")
        if kernel is not None:
            kernel.ints[:] = list(trimmed.shape[2:])
        changes.append(
            {
                "node": node.name or f"node_{index}",
                "op_type": node.op_type,
                "old_shape": list(weight.shape),
                "new_shape": list(trimmed.shape),
                "old_pads": old_pads,
                "new_pads": new_pads,
                "removed_parameters": int(weight.size - trimmed.size),
            }
        )
    return changes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-dir", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--tasks", default="")
    args = parser.parse_args()

    sys.path.insert(0, str(HERE.parent))
    from c_score_common import score_onnx

    owners = _owners()
    requested = {item.strip() for item in args.tasks.split(",") if item.strip()}
    tasks = sorted(requested or {f"task{index:03d}" for index in range(1, 401)})
    args.work_dir.mkdir(parents=True, exist_ok=True)

    promoted = 0
    total_delta_cost = 0
    total_delta_points = 0.0
    opportunity_tasks = 0
    for task in tasks:
        parent_path = args.parent_dir / f"{task}.onnx"
        canonical_path = _canonical(owners[task], task)
        sources = [("parent", parent_path)]
        if canonical_path.is_file() and canonical_path.resolve() != parent_path.resolve():
            sources.append(("canonical", canonical_path))

        built: list[tuple[str, Path, list[dict]]] = []
        for source_name, source_path in sources:
            model = onnx.load(source_path)
            changes = crop_zero_support(model)
            if not changes:
                continue
            candidate_path = args.work_dir / f"{task}_{source_name}.onnx"
            model.producer_name = "ngc_full400_zero_support_crop"
            try:
                onnx.checker.check_model(model, full_check=True)
                inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
                onnx.checker.check_model(inferred, full_check=True)
                onnx.save(inferred, candidate_path)
            except Exception as exc:
                candidate_path.unlink(missing_ok=True)
                print(
                    json.dumps(
                        {
                            "task": task,
                            "source": source_name,
                            "stage": "checker",
                            "rejected": f"{type(exc).__name__}:{exc}",
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
                continue
            built.append((source_name, candidate_path, changes))

        if not built:
            continue
        opportunity_tasks += 1
        parent = score_onnx(task, parent_path, validate_all=True)
        canonical = (
            score_onnx(task, canonical_path, validate_all=True)
            if canonical_path.is_file()
            else None
        )
        current_cost = parent.cost
        if canonical and canonical.ok and canonical.cost is not None:
            current_cost = min(current_cost, canonical.cost)

        winner = None
        for source_name, candidate_path, changes in built:
            candidate = score_onnx(task, candidate_path, validate_all=True)
            valid = bool(
                candidate.ok
                and candidate.examples_checked == candidate.examples_passed
                and candidate.cost is not None
            )
            if valid and (winner is None or candidate.cost < winner[0].cost):
                winner = (candidate, source_name, candidate_path, changes)

        did_promote = bool(winner and winner[0].cost < current_cost)
        if did_promote:
            candidate, source_name, candidate_path, changes = winner
            canonical_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate_path, canonical_path)
            promoted += 1
            delta_cost = int(parent.cost - candidate.cost)
            delta_points = float(candidate.points - parent.points)
            total_delta_cost += delta_cost
            total_delta_points += delta_points
        else:
            candidate = winner[0] if winner else None
            source_name = winner[1] if winner else None
            changes = winner[3] if winner else []
            delta_cost = 0
            delta_points = 0.0

        for _, candidate_path, _ in built:
            candidate_path.unlink(missing_ok=True)
        print(
            json.dumps(
                {
                    "task": task,
                    "source": source_name,
                    "changes": changes,
                    "parent_cost": parent.cost,
                    "canonical_cost_before": canonical.cost if canonical else None,
                    "candidate_cost": candidate.cost if candidate else None,
                    "full_validation": (
                        f"{candidate.examples_passed}/{candidate.examples_checked}"
                        if candidate
                        else "0/0"
                    ),
                    "promoted": did_promote,
                    "delta_cost_vs_parent": delta_cost,
                    "delta_points_vs_parent": delta_points,
                    "canonical_sha": _sha256(canonical_path) if did_promote else None,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )

    print(
        json.dumps(
            {
                "tasks_scanned": len(tasks),
                "opportunity_tasks": opportunity_tasks,
                "promoted": promoted,
                "total_delta_cost_vs_parent": total_delta_cost,
                "total_delta_points_vs_parent": total_delta_points,
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
