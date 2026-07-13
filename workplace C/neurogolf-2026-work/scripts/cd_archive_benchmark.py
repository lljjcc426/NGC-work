from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import onnx


HERE = Path(__file__).resolve()
PROJECT = HERE.parents[1]
WORKPLACE_C = PROJECT.parent
REPO = WORKPLACE_C.parent
DEFAULT_ARCHIVE = PROJECT / "data" / "external" / "neurogolf7300_archive" / "onnx"
DEFAULT_PARENT = Path(
    r"E:/kagglegolf/submissions/candidates/"
    r"GOLF_20260712_102_submission5_plus_c22/onnx"
)
DEFAULT_OUTPUT = WORKPLACE_C / "score_docs" / "41_CD_NEUROGOLF7300_BENCHMARK.csv"


def task_ids() -> list[str]:
    path = REPO / "assignments" / "task_assignment_400.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [
            row["task"]
            for row in csv.DictReader(handle)
            if row.get("owner") in {"C", "D"}
        ]


def graph_signature(path: Path) -> dict[str, object]:
    model = onnx.load(str(path), load_external_data=False)
    operators: dict[str, int] = {}
    for node in model.graph.node:
        operators[node.op_type] = operators.get(node.op_type, 0) + 1
    return {
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "operators": ";".join(f"{key}:{operators[key]}" for key in sorted(operators)),
    }


def score_pair(args: tuple[str, str, str]) -> dict[str, object]:
    task, parent_dir, archive_dir = args
    sys.path.insert(0, str(HERE.parent))
    from c_score_common import score_onnx

    parent = Path(parent_dir) / f"{task}.onnx"
    archive = Path(archive_dir) / f"{task}.onnx"
    row: dict[str, object] = {
        "task": task,
        "parent_path": str(parent),
        "archive_path": str(archive),
        "archive_present": archive.exists(),
    }
    if not parent.exists():
        row["error"] = "missing_parent"
        return row
    try:
        parent_score = score_onnx(task, parent, True)
        row.update({f"parent_{key}": value for key, value in asdict(parent_score).items()})
        row.update({f"parent_graph_{key}": value for key, value in graph_signature(parent).items()})
    except Exception as exc:
        row["error"] = f"parent:{type(exc).__name__}:{exc}"
        return row
    if not archive.exists():
        row["error"] = "missing_archive"
        return row
    try:
        archive_score = score_onnx(task, archive, True)
        row.update({f"archive_{key}": value for key, value in asdict(archive_score).items()})
        row.update({f"archive_graph_{key}": value for key, value in graph_signature(archive).items()})
        if parent_score.cost is not None and archive_score.cost is not None:
            row["archive_delta_cost"] = parent_score.cost - archive_score.cost
            row["archive_lower_cost"] = archive_score.cost < parent_score.cost
        row["archive_candidate_eligible"] = bool(
            archive_score.ok
            and parent_score.cost is not None
            and archive_score.cost is not None
            and archive_score.cost < parent_score.cost
        )
    except Exception as exc:
        row["error"] = f"archive:{type(exc).__name__}:{exc}"
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
    parser.add_argument("--parent-dir", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    tasks = task_ids()
    rows: list[dict[str, object]] = []
    jobs = [(task, str(args.parent_dir), str(args.archive_dir)) for task in tasks]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(score_pair, job): job[0] for job in jobs}
        for index, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {"task": task, "error": f"worker:{type(exc).__name__}:{exc}"}
            rows.append(row)
            rows.sort(key=lambda item: str(item["task"]))
            write_rows(args.output, rows)
            print(
                json.dumps(
                    {
                        "completed": index,
                        "total": len(tasks),
                        "task": task,
                        "eligible": row.get("archive_candidate_eligible", False),
                        "delta_cost": row.get("archive_delta_cost"),
                        "error": row.get("error", ""),
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )

    summary = {
        "tasks": len(tasks),
        "archive_present": sum(bool(row.get("archive_present")) for row in rows),
        "parent_valid": sum(str(row.get("parent_ok", "")).lower() == "true" for row in rows),
        "archive_valid": sum(str(row.get("archive_ok", "")).lower() == "true" for row in rows),
        "archive_lower_and_valid": sum(bool(row.get("archive_candidate_eligible")) for row in rows),
        "errors": sum(bool(row.get("error")) for row in rows),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=True))


if __name__ == "__main__":
    main()
