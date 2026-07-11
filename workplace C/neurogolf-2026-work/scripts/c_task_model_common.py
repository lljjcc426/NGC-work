from __future__ import annotations

import csv
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable, Protocol

from c_score_common import (
    CURRENT_BEST_ONNX_DIR,
    REPO_ROOT,
    SCORE_DOCS,
    WORKPLACE_C,
    points_from_cost,
    score_onnx,
)


TARGET_COST = 148
STATUS_CSV = SCORE_DOCS / "C_20_POINT_STATUS.csv"
STATUS_MD = SCORE_DOCS / "C_20_POINT_STATUS.md"
SINGLE_TASK_ROOT = WORKPLACE_C / "single_task"
TASK_DATA_ROOT = REPO_ROOT / "neurogolf_400_tasks" / "tasks"
STATUS_FIELDS = [
    "task",
    "current_cost",
    "best_cost",
    "points",
    "local_target_met",
    "rule_valid",
    "onnx_valid",
    "examples_passed",
    "examples_total",
    "public_parent_score",
    "public_candidate_score",
    "public_delta",
    "online_verified",
    "artifact_path",
    "blocker",
]


Grid = list[list[int]]


class TaskModel(Protocol):
    TASK_ID: str

    @staticmethod
    def solve(grid: Grid) -> Grid: ...

    @staticmethod
    def build_onnx(output_path: Path) -> Path: ...


@dataclass
class RuleValidation:
    rows: list[dict]
    passed: int
    total: int
    errors: list[str]

    @property
    def ok(self) -> bool:
        return self.total > 0 and self.passed == self.total and not self.errors


def normalize_task(value: str) -> str:
    value = Path(value).stem
    if value.startswith("task"):
        suffix = value[4:]
    else:
        suffix = value
    return f"task{int(suffix):03d}"


def task_dir(task: str) -> Path:
    return SINGLE_TASK_ROOT / normalize_task(task)


def task_model_path(task: str) -> Path:
    return task_dir(task) / "scripts" / "task_model.py"


def candidate_path(task: str) -> Path:
    task = normalize_task(task)
    return task_dir(task) / "onnx" / f"{task}_candidate.onnx"


def baseline_path(task: str, baseline_dir: Path | None = None) -> Path:
    task = normalize_task(task)
    return (baseline_dir or CURRENT_BEST_ONNX_DIR) / f"{task}.onnx"


def load_task(task: str) -> dict:
    task = normalize_task(task)
    return json.loads((TASK_DATA_ROOT / f"{task}.json").read_text(encoding="utf-8"))


def load_task_model(task: str, model_path: Path | None = None) -> ModuleType:
    task = normalize_task(task)
    path = model_path or task_model_path(task)
    if not path.exists():
        raise FileNotFoundError(f"task model not found: {path}")
    spec = importlib.util.spec_from_file_location(f"c_{task}_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load task model: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "solve", None)):
        raise TypeError(f"{path} does not expose solve(grid)")
    if not callable(getattr(module, "build_onnx", None)):
        raise TypeError(f"{path} does not expose build_onnx(output_path)")
    declared = getattr(module, "TASK_ID", task)
    if normalize_task(str(declared)) != task:
        raise ValueError(f"task model declares TASK_ID={declared}, expected {task}")
    return module


def mismatch_count(expected: Grid, predicted: Grid) -> int:
    if len(expected) != len(predicted):
        return sum(len(row) for row in expected) + sum(len(row) for row in predicted)
    mismatch = 0
    for expected_row, predicted_row in zip(expected, predicted):
        if len(expected_row) != len(predicted_row):
            mismatch += max(len(expected_row), len(predicted_row))
            continue
        mismatch += sum(a != b for a, b in zip(expected_row, predicted_row))
    return mismatch


def validate_rule(task: str, solve: Callable[[Grid], Grid]) -> RuleValidation:
    payload = load_task(task)
    rows: list[dict] = []
    errors: list[str] = []
    passed = 0
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(payload.get(split, [])):
            inp = example["input"]
            expected = example["output"]
            error = ""
            try:
                predicted = solve([row[:] for row in inp])
                mismatch = mismatch_count(expected, predicted)
            except Exception as exc:
                predicted = []
                mismatch = sum(len(row) for row in expected)
                error = f"{type(exc).__name__}: {exc}"
                errors.append(f"{split}[{index}]:{error}")
            ok = mismatch == 0 and not error
            passed += int(ok)
            rows.append(
                {
                    "split": split,
                    "index": index,
                    "input_shape": f"{len(inp)}x{len(inp[0]) if inp else 0}",
                    "output_shape": f"{len(expected)}x{len(expected[0]) if expected else 0}",
                    "passed": ok,
                    "mismatch_count": mismatch,
                    "error": error,
                }
            )
    return RuleValidation(rows=rows, passed=passed, total=len(rows), errors=errors)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in rows)


def read_status() -> list[dict[str, str]]:
    if not STATUS_CSV.exists() or not STATUS_CSV.stat().st_size:
        return []
    with STATUS_CSV.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_status(rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda row: normalize_task(row["task"]))
    write_csv(STATUS_CSV, rows, STATUS_FIELDS)
    target_met = sum(str(row.get("local_target_met", "")).lower() == "true" for row in rows)
    online = sum(str(row.get("online_verified", "")).lower() == "true" for row in rows)
    lines = [
        "# C Group 20+ Status",
        "",
        f"- tasks: `{len(rows)}`",
        f"- local_target_met: `{target_met}/{len(rows)}`",
        f"- online_verified: `{online}/{len(rows)}`",
        f"- target_cost: `<= {TARGET_COST}`",
        "",
        "| task | current | best | points | rule | ONNX | 20+ | online | blocker |",
        "| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row.get('current_cost', '')} | {row.get('best_cost', '')} | "
            f"{row.get('points', '')} | {row.get('rule_valid', '')} | {row.get('onnx_valid', '')} | "
            f"{row.get('local_target_met', '')} | {row.get('online_verified', '')} | {row.get('blocker', '')} |"
        )
    STATUS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def upsert_status(row: dict) -> None:
    rows = read_status()
    by_task = {normalize_task(item["task"]): item for item in rows}
    task = normalize_task(row["task"])
    existing = by_task.get(task, {field: "" for field in STATUS_FIELDS})
    existing.update(row)
    existing["task"] = task
    by_task[task] = existing
    write_status(list(by_task.values()))


def status_row_from_result(
    task: str,
    *,
    current_cost: int | float,
    rule_validation: RuleValidation,
    onnx_result,
    artifact: Path,
) -> dict:
    cost = onnx_result.cost if onnx_result.cost is not None else current_cost
    target_met = bool(onnx_result.ok and cost <= TARGET_COST)
    if not rule_validation.ok:
        blocker = "python_rule_failed"
    elif not onnx_result.ok:
        blocker = onnx_result.error or "onnx_validation_failed"
    elif not target_met:
        blocker = f"cost_above_target:{cost}>{TARGET_COST}"
    else:
        blocker = ""
    return {
        "task": normalize_task(task),
        "current_cost": current_cost,
        "best_cost": cost,
        "points": onnx_result.points if onnx_result.points is not None else points_from_cost(cost),
        "local_target_met": str(target_met).lower(),
        "rule_valid": str(rule_validation.ok).lower(),
        "onnx_valid": str(bool(onnx_result.ok)).lower(),
        "examples_passed": onnx_result.examples_passed,
        "examples_total": onnx_result.examples_checked,
        "artifact_path": str(artifact),
        "blocker": blocker,
    }


def score_candidate(task: str, path: Path):
    return score_onnx(normalize_task(task), path, validate_all=True)
