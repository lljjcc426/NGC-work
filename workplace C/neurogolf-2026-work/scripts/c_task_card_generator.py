from __future__ import annotations

from datetime import datetime

from c_score_common import (
    SCORE_DOCS,
    TASK_CARDS,
    TASK_DATA_DIR,
    current_scoreboard,
    ensure_dirs,
    priority_rank,
    task_manifest,
    task_summary,
    write_md,
)


def opportunity_text(row: dict, summary: dict) -> list[str]:
    items = [
        "Run artifact scan against all local public and candidate ONNX sources.",
        "Fully validate any lower-cost artifact on train + test + arc-gen before accepting.",
        "Compare official memory/params split to locate whether memory graph or constants dominate.",
    ]
    if row["shape_class"] == "same_shape":
        items.append("Try same-shape fast path: simplify masks, color comparisons, and identity-preserving branches.")
    else:
        items.append("Trace output shape path first; shape-change tasks need crop/grow logic preservation before compression.")
    if row["color_class"] == "new_output_colors":
        items.append("Audit recolor constants and replace broad per-color logic with narrow constant/color-map paths.")
    else:
        items.append("Exploit input-palette-only constraint; remove unused output color branches and redundant compares.")
    if summary["avg_changed_cell_ratio_same_shape"] and summary["avg_changed_cell_ratio_same_shape"] < 0.2:
        items.append("Changed-cell ratio is low; test sparse mask overlay instead of full-grid recompute.")
    return items


def main() -> None:
    ensure_dirs()
    scoreboard = current_scoreboard()
    cards = []
    for row in task_manifest():
        task = row["task"]
        score = scoreboard.get(task, {})
        summary = task_summary(TASK_DATA_DIR / f"{task}.json")
        deep = row["priority_band"] in {"P0_lt16", "P1_16_16p7"}
        current_cost = score.get("total_cost") or row["cost"]
        current_score = score.get("current_score") or row["points"]
        opp = opportunity_text(row, summary)
        lines = [
            f"# {task} Score Card",
            "",
            f"Generated: {datetime.now().isoformat(timespec='seconds')}",
            "",
            "## Score Priority",
            "",
            f"- priority_band: `{row['priority_band']}`",
            f"- assignment_cost: `{row['cost']}`",
            f"- assignment_points: `{row['points']}`",
            f"- current_cost: `{current_cost}`",
            f"- current_score: `{current_score}`",
            f"- quick depth: `{'deep P0/P1' if deep else 'light P2/P3'}`",
            "",
            "## Why This Task Matters",
            "",
            f"- C track role: `{row['owner_track']}`.",
            f"- Score gap focus: cost compression on current ONNX artifact; no rule rewrite is accepted without full validation.",
            "",
            "## Structure",
            "",
            f"- shape_class: `{row['shape_class']}`",
            f"- size_trend: `{row['size_trend']}`",
            f"- color_class: `{row['color_class']}`",
            f"- train/test/arc-gen: `{summary['train_examples']}/{summary['test_examples']}/{summary['arc_gen_examples']}`",
            f"- input_shapes: `{summary['input_shapes']}`",
            f"- output_shapes: `{summary['output_shapes']}`",
            f"- same_shape_all_examples: `{summary['same_shape_all_examples']}`",
            f"- output_colors_subset_input: `{summary['output_colors_subset_input']}`",
            f"- avg_changed_cell_ratio_same_shape: `{summary['avg_changed_cell_ratio_same_shape']:.4f}`",
            "",
            "## Pattern Understanding",
            "",
        ]
        if row["shape_class"] == "same_shape":
            lines.append("- Same-shape task. Prioritize mask/color logic compression and removal of redundant full-grid branches.")
        elif row["size_trend"] == "shrink":
            lines.append("- Shrink task. Prioritize crop/bounding-box/selection path review before graph-level simplification.")
        else:
            lines.append("- Grow task. Prioritize shape construction and repeat/tile path review before graph-level simplification.")
        if row["color_class"] == "new_output_colors":
            lines.append("- Output introduces colors not always present in input; recolor constants are risk-sensitive.")
        else:
            lines.append("- Output palette is input-contained; unused color creation branches are likely removable.")
        lines += [
            "",
            "## ONNX Compression Opportunities",
            "",
        ]
        lines.extend(f"- {x}" for x in opp)
        lines += [
            "",
            "## Concrete Next Experiments",
            "",
            f"1. `python \"workplace C/neurogolf-2026-work/scripts/c_score_scan_artifacts.py\" --tasks {task} --score-top-n 8 --full-validate`",
            f"2. Inspect best lower-size artifacts in `workplace C/score_docs/artifact_scans/` and compare memory/params split for `{task}`.",
            f"3. If a lower-cost artifact validates, register it with `c_cost_diff_runner.py --task {task} --old-artifact <current> --new-artifact <candidate> --accept-if-better`.",
            "",
            "## Cost Diff",
            "",
            "| attempt | old_cost | new_cost | delta_cost | local_valid | accepted | artifact_path |",
            "| --- | ---: | ---: | ---: | --- | --- | --- |",
            "| pending |  |  |  |  |  |  |",
            "",
            "## Attempts",
            "",
            "- No accepted C-local attempt recorded yet.",
            "",
            "## Next Best Action",
            "",
            "- Run artifact scan and accept the first full-validation lower-cost artifact.",
        ]
        path = TASK_CARDS / f"{task}.md"
        write_md(path, "\n".join(lines))
        cards.append(path)
    summary = [
        "# P0/P1 Experiment Roadmap",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Roadmap: scan artifacts first, then run direct ONNX compression only where no public/local artifact beats current cost.",
        "",
        "| task | priority | first experiment | second experiment | third experiment |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in task_manifest():
        if priority_rank(row["priority_band"]) > 1:
            continue
        task = row["task"]
        summary.append(
            f"| {task} | {row['priority_band']} | full-validate top artifact candidates | inspect memory/params dominant cost | try graph simplify / constant pruning if artifact scan fails |"
        )
    write_md(SCORE_DOCS / "20_P0_P1_EXPERIMENT_ROADMAP.md", "\n".join(summary))
    print(f"generated_cards={len(cards)}")


if __name__ == "__main__":
    main()
