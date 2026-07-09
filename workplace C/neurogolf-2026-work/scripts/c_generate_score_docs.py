from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from c_score_common import (
    CURRENT_BEST_ONNX_DIR,
    KAGGLEGOLF_ROOT,
    SCORE_DOCS,
    ensure_dirs,
    p0_p1_tasks,
    write_csv,
    write_md,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def scorecard() -> dict:
    path = KAGGLEGOLF_ROOT / "reports" / "SCORECARD.md"
    out = {
        "best_public_score": "UNKNOWN",
        "best_exp_id": "GOLF_20260709_101_prvsiyan_7266_72_repro",
        "best_submission_id": "UNKNOWN",
    }
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("Current best public score:"):
                out["best_public_score"] = line.split(":", 1)[1].strip()
            elif line.startswith("Current best exp_id:"):
                out["best_exp_id"] = line.split(":", 1)[1].strip()
            elif line.startswith("Current best submission id:"):
                out["best_submission_id"] = line.split(":", 1)[1].strip()
    return out


def candidate_validation() -> dict:
    path = KAGGLEGOLF_ROOT / "submissions" / "candidates" / "GOLF_20260709_101_prvsiyan_7266_72_repro" / "local_validation.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def top_quick(rows: list[dict[str, str]], n: int = 10) -> list[dict[str, str]]:
    pset = set(p0_p1_tasks())
    filtered = [r for r in rows if r.get("task") in pset]
    return filtered[:n]


def generate() -> None:
    ensure_dirs()
    now = datetime.now().isoformat(timespec="seconds")
    card = scorecard()
    validation = candidate_validation()
    priority = read_csv(SCORE_DOCS / "10_C_SCORE_PRIORITY_TABLE.csv")
    artifact_scored = read_csv(SCORE_DOCS / "artifact_scans" / "c_artifact_scored.csv")
    surgery = read_csv(SCORE_DOCS / "artifact_scans" / "c_surgery_probe_results.csv")
    quick = top_quick(priority, 10)
    accepted_artifacts = [r for r in artifact_scored if str(r.get("accepted", "")).lower() == "true"]
    accepted_surgery = [r for r in surgery if str(r.get("accepted", "")).lower() == "true"]
    accepted_total = len(accepted_artifacts) + len(accepted_surgery)

    # Experiment ledger.
    ledger_rows: list[dict] = []
    for row in artifact_scored:
        if row.get("source_label", "").endswith(".onnx"):
            ledger_rows.append(
                {
                    "task": row.get("task", ""),
                    "attempt_id": "artifact_scan_top5",
                    "method": "public/local_artifact_reuse_full_validate",
                    "old_cost": row.get("current_cost", ""),
                    "new_cost": row.get("cost", ""),
                    "delta_cost": row.get("delta_cost", ""),
                    "old_points": "",
                    "new_points": row.get("points", ""),
                    "delta_points": "",
                    "local_valid": row.get("ok", ""),
                    "artifact_path": row.get("source_label", ""),
                    "accepted": row.get("accepted", ""),
                    "notes": row.get("error", ""),
                }
            )
    for row in surgery:
        if row.get("strategy") == "baseline":
            continue
        ledger_rows.append(
            {
                "task": row.get("task", ""),
                "attempt_id": f"surgery_{row.get('strategy','')}",
                "method": "onnxoptimizer/onnxsim_full_validate",
                "old_cost": row.get("old_cost", ""),
                "new_cost": row.get("new_cost", ""),
                "delta_cost": row.get("delta_cost", ""),
                "old_points": row.get("old_points", ""),
                "new_points": row.get("new_points", ""),
                "delta_points": "",
                "local_valid": row.get("ok", ""),
                "artifact_path": row.get("artifact_path", ""),
                "accepted": row.get("accepted", ""),
                "notes": row.get("notes", ""),
            }
        )
    ledger_fields = [
        "task",
        "attempt_id",
        "method",
        "old_cost",
        "new_cost",
        "delta_cost",
        "old_points",
        "new_points",
        "delta_points",
        "local_valid",
        "artifact_path",
        "accepted",
        "notes",
    ]
    write_csv(SCORE_DOCS / "30_SCORE_EXPERIMENT_LEDGER.csv", ledger_rows, ledger_fields)

    ledger_md = [
        "# Score Experiment Ledger",
        "",
        f"Generated: {now}",
        "",
        f"- artifact scan rows: `{len(artifact_scored)}`",
        f"- surgery probe rows: `{len(surgery)}`",
        f"- accepted improvements: `{accepted_total}`",
        "",
        "| task | attempt | old_cost | new_cost | delta_cost | valid | accepted | artifact |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in ledger_rows:
        ledger_md.append(
            f"| {row['task']} | {row['attempt_id']} | {row['old_cost']} | {row['new_cost']} | "
            f"{row['delta_cost']} | {row['local_valid']} | {row['accepted']} | `{row['artifact_path']}` |"
        )
    write_md(SCORE_DOCS / "30_SCORE_EXPERIMENT_LEDGER.md", "\n".join(ledger_md))

    # Candidate register.
    candidate_rows = [
        {
            "candidate_id": "GOLF_20260709_101_prvsiyan_7266_72_repro",
            "created_at": "2026-07-09",
            "included_tasks": "400 baseline tasks",
            "expected_delta_cost": "0 from this C run",
            "validation_status": "STRUCTURE_400; examples_passed=1197; known task148 structural warning in local validator",
            "file_count_400": validation.get("file_count", ""),
            "missing_task_count": validation.get("missing_task_count", ""),
            "artifact_path": str(CURRENT_BEST_ONNX_DIR),
            "public_lb": card["best_public_score"],
            "notes": "Current external baseline; no C accepted replacement generated in this run.",
        }
    ]
    write_csv(SCORE_DOCS / "31_CANDIDATE_SUBMISSION_REGISTER.csv", candidate_rows)
    write_md(
        SCORE_DOCS / "31_CANDIDATE_SUBMISSION_REGISTER.md",
        "\n".join(
            [
                "# Candidate Submission Register",
                "",
                f"Generated: {now}",
                "",
                "| candidate_id | file_count_400 | missing_task_count | expected_delta_cost | public_lb | status |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
                f"| {candidate_rows[0]['candidate_id']} | {candidate_rows[0]['file_count_400']} | "
                f"{candidate_rows[0]['missing_task_count']} | {candidate_rows[0]['expected_delta_cost']} | "
                f"{candidate_rows[0]['public_lb']} | no new C accepted replacement |",
                "",
                "No new candidate package was built from this C run because `accepted_improvements=0`.",
            ]
        ),
    )

    # State/dashboard.
    state_lines = [
        "# Current C Score Work State",
        "",
        f"Generated: {now}",
        "",
        "- C group tasks identified: 67 primary tasks.",
        f"- P0/P1 tasks identified: {len(p0_p1_tasks())} (`{', '.join(p0_p1_tasks())}`).",
        "- P0/P1 task cards: 67 generated; P0/P1 cards include full validate actions and concrete experiments.",
        f"- ONNX artifacts indexed: `{len(read_csv(SCORE_DOCS / 'artifact_scans' / 'c_artifact_index.csv'))}` P0/P1 unique local artifacts.",
        f"- Officially scored artifact candidates: `{len(artifact_scored)}`; accepted: `{len(accepted_artifacts)}`.",
        f"- ONNX surgery probes: `{len(surgery)}` rows; accepted: `{len(accepted_surgery)}`.",
        "- Cost command exists: `c_score_scan_artifacts.py` and `c_cost_diff_runner.py` use local official `neurogolf_utils.py`.",
        "- Validator exists: `c_validate_candidate.py` checks 400-task candidate completeness.",
        f"- Public notebook baseline: `{card['best_exp_id']}` public LB `{card['best_public_score']}`.",
        "- Current highest-yield next route: write dedicated compact builders for task158/task286/task054/task364 instead of generic graph cleanup.",
    ]
    write_md(SCORE_DOCS / "00_CURRENT_SCORE_WORK_STATE.md", "\n".join(state_lines))

    start_here = [
        "# Score First Start Here",
        "",
        f"Generated: {now}",
        "",
        "Use this workspace for C group ONNX-equivalent compression experiments only. Do not commit `.onnx`, token, zip, or Kaggle output files.",
        "",
        "## Fast Commands",
        "",
        "```powershell",
        "python \"workplace C\\neurogolf-2026-work\\scripts\\c_quick_win_scan.py\"",
        "python \"workplace C\\neurogolf-2026-work\\scripts\\c_score_scan_artifacts.py\" --tasks P0P1 --score-top-n 5 --full-validate",
        "python \"workplace C\\neurogolf-2026-work\\scripts\\c_onnx_surgery_probe.py\" --tasks P0P1 --strategies optimizer,sim,optimizer_sim --full-validate",
        "python \"workplace C\\neurogolf-2026-work\\scripts\\c_validate_candidate.py\" --candidate-dir \"E:\\kagglegolf\\submissions\\candidates\\GOLF_20260709_101_prvsiyan_7266_72_repro\\onnx\"",
        "```",
    ]
    write_md(SCORE_DOCS / "00_SCORE_FIRST_START_HERE.md", "\n".join(start_here))

    dash = [
        "# Score Status Dashboard",
        "",
        f"Generated: {now}",
        "",
        f"- Current reproduced public LB: `{card['best_public_score']}`.",
        f"- C P0/P1 artifact reuse accepted improvements: `{len(accepted_artifacts)}`.",
        f"- C P0/P1 surgery accepted improvements: `{len(accepted_surgery)}`.",
        f"- Candidate submissions produced by this run: `0`.",
        f"- Candidate structure check baseline: file_count `{validation.get('file_count','UNKNOWN')}`, missing `{validation.get('missing_task_count','UNKNOWN')}`.",
        "",
        "## Quick-Win Top 10",
        "",
        "| rank | task | priority | current_cost | current_points | shape | color |",
        "| ---: | --- | --- | ---: | ---: | --- | --- |",
    ]
    for i, row in enumerate(quick, 1):
        dash.append(
            f"| {i} | {row.get('task')} | {row.get('priority_band')} | {row.get('current_cost')} | "
            f"{row.get('current_points')} | {row.get('shape_class')}/{row.get('size_trend')} | {row.get('color_class')} |"
        )
    write_md(SCORE_DOCS / "01_SCORE_STATUS_DASHBOARD.md", "\n".join(dash))

    command_docs = [
        "# Cost Commands",
        "",
        f"Generated: {now}",
        "",
        "Primary official scoring path:",
        "",
        "```powershell",
        "python \"workplace C\\neurogolf-2026-work\\scripts\\c_score_scan_artifacts.py\" --tasks P0P1 --score-top-n 5 --full-validate",
        "python \"workplace C\\neurogolf-2026-work\\scripts\\c_cost_diff_runner.py\" --task task158 --old-artifact <old.onnx> --new-artifact <new.onnx> --method <method>",
        "```",
        "",
        "The scripts import `E:/kagglegolf/data/raw/neurogolf-2026/neurogolf_utils/neurogolf_utils.py` and compute `cost=memory+params` after local example validation.",
    ]
    write_md(SCORE_DOCS / "25_COST_COMMANDS.md", "\n".join(command_docs))

    validator_docs = [
        "# Validator Commands",
        "",
        f"Generated: {now}",
        "",
        "```powershell",
        "python \"workplace C\\neurogolf-2026-work\\scripts\\c_validate_candidate.py\" --candidate-dir <candidate_onnx_dir>",
        "python \"workplace C\\neurogolf-2026-work\\scripts\\c_validate_candidate.py\" --candidate-zip <submission.zip>",
        "```",
        "",
        "Validation here checks 400-task structural completeness only. Cost acceptance requires the scorer scripts with `--full-validate`.",
    ]
    write_md(SCORE_DOCS / "26_VALIDATOR_COMMANDS.md", "\n".join(validator_docs))

    playbook = [
        "# ONNX Score Improvement Playbook",
        "",
        f"Generated: {now}",
        "",
        "Current generic cleanup result: no cost decrease on P0/P1. Future score work should avoid repeating optimizer/sim-only passes unless the source graph changes.",
        "",
        "Highest-value tactics:",
        "",
        "1. Dedicated compact builder for `task158`: detect motif/template and emit smaller same-shape fill network.",
        "2. Dedicated compact builder for `task286`: replace 2393-node iterative graph with a compact propagation primitive if the color-fill rule is formalized.",
        "3. Dedicated compact builder for `task054`: line/cross propagation from marker pixel, not generic graph simplification.",
        "4. Dedicated compact builder for `task364`: classify connected shape glyphs into colors 1/2/6 with a component/neighborhood network.",
        "5. Mine `prvsiyan_7266_72/output/visualizations` for task-level explanations before coding more ONNX.",
    ]
    write_md(SCORE_DOCS / "21_ONNX_SCORE_IMPROVEMENT_PLAYBOOK.md", "\n".join(playbook))

    protocol = [
        "# Cost Diff And Acceptance Protocol",
        "",
        f"Generated: {now}",
        "",
        "Accepted requires all of:",
        "",
        "1. Full local example validation passes.",
        "2. `new_cost < old_cost` under official `neurogolf_utils.py` scoring.",
        "3. Artifact path exists locally.",
        "4. Experiment row is written to `30_SCORE_EXPERIMENT_LEDGER.csv`.",
        "",
        "Exploratory rows with missing cost or failed validation are never marked accepted.",
    ]
    write_md(SCORE_DOCS / "22_COST_DIFF_AND_ACCEPTANCE_PROTOCOL.md", "\n".join(protocol))

    val_protocol = [
        "# Score Validation Protocol",
        "",
        f"Generated: {now}",
        "",
        "- Candidate package completeness: `c_validate_candidate.py`.",
        "- Per-task functional/cost validation: `c_score_scan_artifacts.py` or `c_cost_diff_runner.py` with `--full-validate`.",
        "- Submission requires explicit user confirmation and is not triggered by these scripts.",
    ]
    write_md(SCORE_DOCS / "23_SCORE_VALIDATION_PROTOCOL.md", "\n".join(val_protocol))

    public_sources = [
        "# Public Score Intel",
        "",
        f"Generated: {now}",
        "",
        "| source | title | claimed_score | verified_score | C relevance | usable action | trust |",
        "| --- | --- | ---: | ---: | --- | --- | --- |",
        "| local submission history | prvsiyan/neurogolf-7266-72-w-visualizations | 7266.72 | 7266.72 | current baseline for all C P0/P1 | compare/replace only if lower cost | high |",
        "| local Kaggle kernel list | seddiktrk/neurogolf-2026-all-graph-surgeries | UNKNOWN | UNKNOWN | graph-surgery ideas; scanned artifacts were worse on P0/P1 | mine technique, not direct replace | medium |",
        "| local Kaggle kernel list | ryosukeshiroshita/neurogolf-7266-48-github-com-qurore-kaggloop | 7266.48 | UNKNOWN | near-current public solution | inspect task-level differences if cloned | medium |",
        "| local public bundles | kojimar/jsrdcht/vyanktesh/beicicc/afr1ste outputs | mixed | local cost-scanned | many C artifacts present but higher cost or invalid | no direct C P0/P1 replace | high for local cost result |",
    ]
    write_md(SCORE_DOCS / "40_PUBLIC_SCORE_INTEL.md", "\n".join(public_sources))

    write_md(
        SCORE_DOCS / "41_PUBLIC_ONNX_BASELINES.md",
        "\n".join(
            [
                "# Public ONNX Baselines",
                "",
                f"Generated: {now}",
                "",
                f"- Active baseline ONNX dir: `{CURRENT_BEST_ONNX_DIR}`.",
                "- P0/P1 artifact scan scored 73 local public/local artifacts.",
                "- No public/local artifact beat the current prvsiyan 7266.72 task cost on any C P0/P1 task.",
                "- Full details: `artifact_scans/c_artifact_scored.csv` and `24_ONNX_ARTIFACT_INDEX.md`.",
            ]
        ),
    )

    coverage_lines = [
        "# Public Task Coverage C",
        "",
        f"Generated: {now}",
        "",
        "| task | current_cost | best_scanned_nonbaseline_cost | accepted |",
        "| --- | ---: | ---: | --- |",
    ]
    by_task: dict[str, list[dict[str, str]]] = {}
    for row in artifact_scored:
        by_task.setdefault(row.get("task", ""), []).append(row)
    for task in p0_p1_tasks():
        rows = [r for r in by_task.get(task, []) if r.get("source_label", "").find("prvsiyan_7266") < 0 and r.get("cost")]
        best = min((float(r["cost"]) for r in rows), default=None)
        current = next((r.get("current_cost") for r in by_task.get(task, []) if r.get("current_cost")), "")
        coverage_lines.append(f"| {task} | {current} | {'' if best is None else int(best)} | false |")
    write_md(SCORE_DOCS / "42_PUBLIC_TASK_COVERAGE_C.md", "\n".join(coverage_lines))

    repro = [
        "# Public Reproduction Plan",
        "",
        f"Generated: {now}",
        "",
        "The public 7266.72 baseline is already reproduced locally and submitted once under the configured Kaggle user. The next reproduction target is not another full notebook run; it is task-level diff extraction against near-7266.48/7266.72 public artifacts.",
        "",
        "Next reproduction actions:",
        "",
        "1. Extract Ryosuke/KaggLoop artifact set and run `c_score_scan_artifacts.py --tasks P0P1 --score-top-n 20 --full-validate` if available locally.",
        "2. Compare prvsiyan task158/task286/task054 graphs with public Python solutions and visualization notes.",
        "3. Build one dedicated compact builder at a time, starting with task364 or task054.",
    ]
    write_md(SCORE_DOCS / "43_PUBLIC_REPRODUCTION_PLAN.md", "\n".join(repro))

    next_tasks = [
        "# Next Score Research Tasks",
        "",
        f"Generated: {now}",
        "",
        "1. `task158`: formalize motif-copy rule and design a smaller same-shape ONNX; current cost 28483.",
        "2. `task286`: profile 2393-node current graph and identify repeated propagation blocks; current cost 26909.",
        "3. `task054`: derive marker-driven row/column propagation rule; current cost 25394.",
        "4. `task364`: implement component-shape classifier for colors 1/2/6; current cost 14642.",
        "5. `task349`: understand rectangle/connector fill between 9-blocks; current cost 14892.",
    ]
    write_md(SCORE_DOCS / "50_NEXT_SCORE_RESEARCH_TASKS.md", "\n".join(next_tasks))

    workflow = [
        "# Score First Workflow",
        "",
        f"Generated: {now}",
        "",
        "1. Pick a P0 task from `11_C_QUICK_WIN_SCAN.md`.",
        "2. Inspect task card and local visualization.",
        "3. Build a dedicated ONNX generator into `E:/kagglegolf` or ignored `workplace C/artifacts`.",
        "4. Run `c_cost_diff_runner.py` against current prvsiyan artifact.",
        "5. Only if accepted, update `31_CANDIDATE_SUBMISSION_REGISTER.md` and build a 400-file candidate outside Git-tracked artifacts.",
    ]
    write_md(SCORE_DOCS / "60_SCORE_FIRST_WORKFLOW.md", "\n".join(workflow))

    report = [
        "# Score Work Report",
        "",
        f"Generated: {now}",
        "",
        "## Modes",
        "",
        "- Mode A: ran P0/P1 artifact reuse scan and official full-validation cost scoring.",
        "- Mode B: created reusable artifact scanner, cost diff runner, candidate validator, task card generator, and surgery probe scripts.",
        "- Mode C: summarized local public notebook/bundle intel, centered on the verified prvsiyan 7266.72 baseline.",
        "- Mode D: candidate register updated, but no new candidate package was built because there were zero accepted C replacements.",
        "- Mode E: minimal score docs/task cards generated to support next experiments.",
        "",
        "## Direct Score Attempts",
        "",
        f"- P0/P1 artifacts indexed: `{len(read_csv(SCORE_DOCS / 'artifact_scans' / 'c_artifact_index.csv'))}`.",
        f"- Artifact rows full-scored: `{len(artifact_scored)}`; accepted: `{len(accepted_artifacts)}`.",
        f"- Surgery rows full-scored: `{len(surgery)}`; accepted: `{len(accepted_surgery)}`.",
        "- Generic optimizer/simplifier passes did not reduce official cost; several files became larger on disk but cost stayed identical.",
        "",
        "## Cost Results",
        "",
        "| source | scored_rows | accepted | best_delta_cost |",
        "| --- | ---: | ---: | ---: |",
        f"| artifact_scan_top5 | {len(artifact_scored)} | {len(accepted_artifacts)} | 0 |",
        f"| onnx_surgery_probe | {len(surgery)} | {len(accepted_surgery)} | 0 |",
        "",
        "## Quick-Win Top 10",
        "",
        "| rank | task | priority | current_cost | current_points |",
        "| ---: | --- | --- | ---: | ---: |",
    ]
    for i, row in enumerate(quick, 1):
        report.append(f"| {i} | {row.get('task')} | {row.get('priority_band')} | {row.get('current_cost')} | {row.get('current_points')} |")
    report.extend(
        [
            "",
            "## Next 5 Experiments",
            "",
            "1. Dedicated compact builder for `task158` motif-copy/fill.",
            "2. Dedicated compact builder for `task286` repeated propagation.",
            "3. Dedicated compact builder for `task054` marker-driven cross/line overwrite.",
            "4. Dedicated compact component-shape classifier for `task364`.",
            "5. Mine prvsiyan visualizations and KaggLoop 7266.48 for task-level graph differences before more generic surgery.",
            "",
            "## Blockers",
            "",
            "- No accepted lower-cost artifact found in local public artifact pool for C P0/P1.",
            "- Existing current graphs are already optimizer-stable for official cost; generic simplification is not enough.",
            "- Real score improvement now needs task-specific ONNX construction, not more template documentation.",
            "",
            "## Git",
            "",
            "Git checkpoint attempted after this report generation; see final assistant summary for commit status.",
        ]
    )
    write_md(SCORE_DOCS / "99_SCORE_WORK_REPORT.md", "\n".join(report))


if __name__ == "__main__":
    generate()
    print(SCORE_DOCS / "99_SCORE_WORK_REPORT.md")
