from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from c_score_common import (
    ARTIFACTS_DIR,
    CURRENT_BEST_ONNX_DIR,
    SCORE_DOCS,
    ensure_dirs,
    current_scoreboard,
    iter_task_artifacts,
    p0_p1_tasks,
    rel_to_kagglegolf,
    score_onnx,
    score_result_row,
    sha256_file,
    task_manifest,
    write_csv,
    write_md,
)


def parse_tasks(value: str) -> list[str]:
    if value.upper() == "P0P1":
        return p0_p1_tasks()
    if value.upper() == "ALLC":
        return [r["task"] for r in task_manifest()]
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="P0P1", help="P0P1, ALLC, or comma-separated task IDs")
    parser.add_argument("--score-top-n", type=int, default=0, help="Officially score the N smallest unique artifacts per task plus current baseline")
    parser.add_argument("--max-examples", type=int, default=0, help="Limit validation examples; 0 means full validation when --full-validate")
    parser.add_argument("--full-validate", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    tasks = parse_tasks(args.tasks)
    scoreboard = current_scoreboard()
    artifact_rows = []
    scored_rows = []
    accepted_rows = []

    for task in tasks:
        current = scoreboard.get(task, {})
        current_cost = float(current.get("total_cost")) if current.get("total_cost") else None
        current_path = CURRENT_BEST_ONNX_DIR / f"{task}.onnx"
        paths = iter_task_artifacts(task)
        if current_path.exists() and current_path not in paths:
            paths.insert(0, current_path)
        seen = set()
        unique = []
        for p in paths:
            try:
                h = sha256_file(p)
            except Exception:
                continue
            if h in seen:
                continue
            seen.add(h)
            unique.append((p.stat().st_size, p, h))
            artifact_rows.append(
                {
                    "task": task,
                    "path": str(p),
                    "source_label": rel_to_kagglegolf(p),
                    "file_size": p.stat().st_size,
                    "sha256": h,
                    "current_cost": current_cost if current_cost is not None else "",
                }
            )
        unique = sorted(unique, key=lambda x: (x[0], str(x[1]).lower()))
        score_targets = []
        if current_path.exists():
            score_targets.append(current_path)
        if args.score_top_n:
            for _, p, _ in unique[: args.score_top_n]:
                if p not in score_targets:
                    score_targets.append(p)
        for p in score_targets:
            result = score_onnx(task, p, validate_all=args.full_validate, max_examples=args.max_examples)
            row = score_result_row(result, current_cost=current_cost, source_label=rel_to_kagglegolf(p))
            scored_rows.append(row)
            if row["accepted"]:
                accepted_rows.append(row)

    scan_dir = SCORE_DOCS / "artifact_scans"
    write_csv(scan_dir / "c_artifact_index.csv", artifact_rows)
    if scored_rows:
        write_csv(scan_dir / "c_artifact_scored.csv", scored_rows)
    if accepted_rows:
        write_csv(scan_dir / "c_accepted_improvements.csv", accepted_rows)
        write_csv(ARTIFACTS_DIR / "accepted_improvements.csv", accepted_rows)

    lines = [
        "# ONNX Artifact Index",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Tasks scanned: `{', '.join(tasks)}`",
        f"Artifact rows: `{len(artifact_rows)}`",
        f"Officially scored rows: `{len(scored_rows)}`",
        f"Accepted lower-cost rows: `{len(accepted_rows)}`",
        "",
        "Accepted means full local validation/cost pass and `new_cost < current_cost`.",
        "",
        "## Accepted Improvements",
        "",
        "| task | current_cost | new_cost | delta_cost | points | artifact |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    if accepted_rows:
        for r in sorted(accepted_rows, key=lambda x: (-float(x["delta_cost"]), x["task"])):
            lines.append(
                f"| {r['task']} | {r['current_cost']} | {r['cost']} | {r['delta_cost']} | {r['points']} | `{r['source_label']}` |"
            )
    else:
        lines.append("| none |  |  |  |  |  |")
    lines += [
        "",
        "## Top Scored Candidates",
        "",
        "| task | ok | cost | current_cost | delta_cost | examples | file_size | artifact | error |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for r in sorted(scored_rows, key=lambda x: (x["task"], x["cost"] if x["cost"] not in ("", None) else 10**18))[:200]:
        lines.append(
            f"| {r['task']} | {r['ok']} | {r['cost']} | {r['current_cost']} | {r['delta_cost']} | {r['examples_checked']} | {r['file_size']} | `{r['source_label']}` | `{str(r['error'])[:80]}` |"
        )
    write_md(SCORE_DOCS / "24_ONNX_ARTIFACT_INDEX.md", "\n".join(lines))
    print(SCORE_DOCS / "24_ONNX_ARTIFACT_INDEX.md")
    print(f"accepted={len(accepted_rows)} scored={len(scored_rows)} artifacts={len(artifact_rows)}")


if __name__ == "__main__":
    main()
