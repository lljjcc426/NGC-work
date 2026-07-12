from __future__ import annotations

import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
WORKPLACE_C = PROJECT_DIR.parent
REPO_ROOT = WORKPLACE_C.parent
SCORE_DOCS = WORKPLACE_C / "score_docs"
TASK_CARDS = SCORE_DOCS / "task_cards"
ARTIFACTS_DIR = WORKPLACE_C / "artifacts"
TASK_DATA_DIR = REPO_ROOT / "neurogolf_400_tasks" / "tasks"
C_TASK_DIR = WORKPLACE_C / "tasks"
KAGGLEGOLF_ROOT = Path(os.environ.get("KAGGLEGOLF_ROOT", r"E:/kagglegolf"))
CURRENT_BEST_ONNX_DIR = (
    KAGGLEGOLF_ROOT
    / "submissions"
    / "candidates"
    / "GOLF_20260709_101_prvsiyan_7266_72_repro"
    / "onnx"
)
CURRENT_SCOREBOARD = KAGGLEGOLF_ROOT / "data" / "neurogolf_task_table" / "task_scoreboard.csv"
OFFICIAL_UTILS = (
    KAGGLEGOLF_ROOT
    / "data"
    / "raw"
    / "neurogolf-2026"
    / "neurogolf_utils"
    / "neurogolf_utils.py"
)

P0_P1_PRIORITY = [
    "task158",
    "task286",
    "task054",
    "task364",
    "task349",
    "task077",
    "task096",
    "task009",
    "task383",
    "task382",
    "task278",
    "task165",
    "task378",
    "task132",
]


def ensure_dirs() -> None:
    SCORE_DOCS.mkdir(parents=True, exist_ok=True)
    TASK_CARDS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (SCORE_DOCS / "artifact_scans").mkdir(parents=True, exist_ok=True)
    (SCORE_DOCS / "experiments").mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def task_manifest() -> list[dict[str, str]]:
    rows = read_csv(WORKPLACE_C / "task_manifest_C.csv")
    return sorted(rows, key=lambda r: (priority_rank(r["priority_band"]), -float(r["cost"]), r["task"]))


def current_scoreboard() -> dict[str, dict[str, str]]:
    if not CURRENT_SCOREBOARD.exists():
        return {}
    rows = read_csv(CURRENT_SCOREBOARD)
    return {r["task_id"]: r for r in rows}


def task_index() -> dict[str, dict[str, str]]:
    rows = read_csv(REPO_ROOT / "neurogolf_400_tasks" / "task_index.csv")
    return {r["task"]: r for r in rows}


def priority_rank(priority_band: str) -> int:
    return {
        "P0_lt16": 0,
        "P1_16_16p7": 1,
        "P2_16p7_17p5": 2,
        "P3_ge17p5": 3,
    }.get(priority_band, 9)


def p0_p1_tasks() -> list[str]:
    rows = task_manifest()
    found = [r["task"] for r in rows if r["priority_band"] in {"P0_lt16", "P1_16_16p7"}]
    ordered = [t for t in P0_P1_PRIORITY if t in found]
    ordered.extend(t for t in found if t not in ordered)
    return ordered


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_roots() -> list[Path]:
    roots = [
        CURRENT_BEST_ONNX_DIR,
        KAGGLEGOLF_ROOT / "data" / "external",
        KAGGLEGOLF_ROOT / "data" / "interim",
        KAGGLEGOLF_ROOT / "data" / "kaggle_code_single_task" / "extracted_onnx",
        KAGGLEGOLF_ROOT / "task_bank" / "tasks",
        KAGGLEGOLF_ROOT / "submissions" / "candidates",
        KAGGLEGOLF_ROOT / "submissions" / "best",
        KAGGLEGOLF_ROOT / "external" / "public_notebooks",
    ]
    return [p for p in roots if p.exists()]


def iter_task_artifacts(task: str, roots: Iterable[Path] | None = None) -> list[Path]:
    paths: list[Path] = []
    for root in roots or artifact_roots():
        if root.is_file():
            continue
        paths.extend(root.rglob(f"{task}.onnx"))
    return sorted(set(paths), key=lambda p: (p.stat().st_size, str(p).lower()))


def rel_to_kagglegolf(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(KAGGLEGOLF_ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path)


def load_official_utils():
    if not OFFICIAL_UTILS.exists():
        raise FileNotFoundError(f"official neurogolf_utils.py not found: {OFFICIAL_UTILS}")
    spec = importlib.util.spec_from_file_location("ngc_neurogolf_utils", OFFICIAL_UTILS)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load official utils from {OFFICIAL_UTILS}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._NEUROGOLF_DIR = str((KAGGLEGOLF_ROOT / "data" / "raw" / "neurogolf-2026").resolve()).replace("\\", "/") + "/"
    return module


def points_from_cost(cost: float | int | None) -> float | None:
    if cost is None:
        return None
    return max(1.0, 25.0 - math.log(max(1.0, float(cost))))


@dataclass
class ScoreResult:
    task: str
    path: str
    ok: bool
    valid_all_checked: bool
    examples_checked: int
    examples_passed: int
    examples_failed: int
    memory: int | None
    params: int | None
    cost: int | None
    points: float | None
    file_size: int
    sha256: str
    error: str = ""


def score_onnx(task: str, onnx_path: Path, validate_all: bool = True, max_examples: int = 0) -> ScoreResult:
    utils = load_official_utils()
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
    except Exception as exc:
        return ScoreResult(task, str(onnx_path), False, False, 0, 0, 0, None, None, None, None, onnx_path.stat().st_size, sha256_file(onnx_path), f"import_error:{exc}")

    checked = passed = failed = 0
    memory = params = cost = None
    try:
        model = onnx.load(str(onnx_path))
        sanitized = utils.sanitize_model(model)
        if sanitized is None:
            raise RuntimeError("sanitize_model returned None")
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        with tempfile.TemporaryDirectory(prefix="ngc_c_score_") as tmp:
            options.profile_file_prefix = str(Path(tmp) / task)
            session = ort.InferenceSession(
                sanitized.SerializeToString(),
                options,
                providers=["CPUExecutionProvider"],
            )
            examples = json.loads((TASK_DATA_DIR / f"{task}.json").read_text(encoding="utf-8"))
            pairs = list(examples.get("train", [])) + list(examples.get("test", [])) + list(examples.get("arc-gen", []))
            if max_examples:
                pairs = pairs[:max_examples]
            if not validate_all and max_examples == 0:
                pairs = pairs[:3]
            for ex in pairs:
                arrays = utils.convert_to_numpy(ex)
                if arrays is None:
                    continue
                checked += 1
                try:
                    out = utils.run_network(session, arrays["input"])
                    if np.array_equal(out, arrays["output"]):
                        passed += 1
                    else:
                        failed += 1
                except Exception:
                    failed += 1
                    if not validate_all:
                        break
            trace = session.end_profiling()
            memory, params = utils.score_network(sanitized, trace)
            # ORT can retain the profiling JSON handle on Windows until the
            # session is finalized, which makes TemporaryDirectory cleanup
            # fail even after end_profiling() returned successfully.
            del session
            gc.collect()
        if memory is not None and params is not None:
            cost = int(memory + params)
        ok = checked > 0 and failed == 0 and cost is not None
        return ScoreResult(
            task=task,
            path=str(onnx_path),
            ok=ok,
            valid_all_checked=validate_all and max_examples == 0,
            examples_checked=checked,
            examples_passed=passed,
            examples_failed=failed,
            memory=int(memory) if memory is not None else None,
            params=int(params) if params is not None else None,
            cost=cost,
            points=points_from_cost(cost),
            file_size=onnx_path.stat().st_size,
            sha256=sha256_file(onnx_path),
        )
    except Exception as exc:
        return ScoreResult(
            task=task,
            path=str(onnx_path),
            ok=False,
            valid_all_checked=validate_all and max_examples == 0,
            examples_checked=checked,
            examples_passed=passed,
            examples_failed=failed,
            memory=memory,
            params=params,
            cost=cost,
            points=points_from_cost(cost),
            file_size=onnx_path.stat().st_size if onnx_path.exists() else 0,
            sha256=sha256_file(onnx_path) if onnx_path.exists() else "",
            error=f"{type(exc).__name__}:{exc}",
        )


def score_result_row(result: ScoreResult, current_cost: float | None = None, source_label: str = "") -> dict:
    row = asdict(result)
    row["source_label"] = source_label or rel_to_kagglegolf(Path(result.path))
    row["current_cost"] = current_cost if current_cost is not None else ""
    row["delta_cost"] = (float(current_cost) - result.cost) if current_cost is not None and result.cost is not None else ""
    row["accepted"] = bool(result.ok and current_cost is not None and result.cost is not None and result.cost < current_cost)
    return row


def task_summary(task_path: Path) -> dict:
    data = json.loads(task_path.read_text(encoding="utf-8"))
    pairs = []
    for split in ("train", "test", "arc-gen"):
        for item in data.get(split, []):
            pairs.append((split, item))
    input_shapes: set[str] = set()
    output_shapes: set[str] = set()
    output_subset = True
    same_shape = True
    changed_ratios: list[float] = []
    for _, item in pairs:
        inp = item["input"]
        out = item["output"]
        ishape = f"{len(inp)}x{len(inp[0]) if inp else 0}"
        oshape = f"{len(out)}x{len(out[0]) if out else 0}"
        input_shapes.add(ishape)
        output_shapes.add(oshape)
        if ishape != oshape:
            same_shape = False
        in_colors = {v for row in inp for v in row}
        out_colors = {v for row in out for v in row}
        if not out_colors.issubset(in_colors):
            output_subset = False
        if ishape == oshape:
            total = len(inp) * len(inp[0]) if inp else 0
            changed = sum(1 for r in range(len(inp)) for c in range(len(inp[0])) if inp[r][c] != out[r][c])
            if total:
                changed_ratios.append(changed / total)
    avg_changed = sum(changed_ratios) / len(changed_ratios) if changed_ratios else 0.0
    return {
        "input_shapes": ";".join(sorted(input_shapes)),
        "output_shapes": ";".join(sorted(output_shapes)),
        "same_shape_all_examples": same_shape,
        "output_colors_subset_input": output_subset,
        "avg_changed_cell_ratio_same_shape": avg_changed,
        "train_examples": len(data.get("train", [])),
        "test_examples": len(data.get("test", [])),
        "arc_gen_examples": len(data.get("arc-gen", [])),
    }
