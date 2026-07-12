from __future__ import annotations

import csv
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
WORKPLACE = REPO / "workplace C"
SINGLE = WORKPLACE / "single_task"
OUTPUT_CSV = WORKPLACE / "score_docs" / "C_DEEP_MODEL_STATUS.csv"
OUTPUT_MD = WORKPLACE / "score_docs" / "C_DEEP_MODEL_STATUS.md"


def main() -> None:
    with (WORKPLACE / "task_manifest_C.csv").open(newline="", encoding="utf-8-sig") as handle:
        manifest = list(csv.DictReader(handle))

    rows = []
    for source in manifest:
        task = source["task"]
        root = SINGLE / task
        modeling = root / "reports" / "modeling.md"
        cost_diff = root / "reports" / "cost_diff.csv"
        builders = sorted((root / "scripts").glob("build*.py")) if (root / "scripts").exists() else []
        candidates = sorted((root / "onnx").glob("*.onnx")) if (root / "onnx").exists() else []
        complete = modeling.exists() and cost_diff.exists() and bool(builders) and bool(candidates)
        rows.append(
            {
                "task": task,
                "priority": source["priority_band"],
                "baseline_cost": source["cost"],
                "deep_model_complete": str(complete).lower(),
                "modeling_report": str(modeling.exists()).lower(),
                "cost_diff": str(cost_diff.exists()).lower(),
                "builder_count": len(builders),
                "candidate_count": len(candidates),
            }
        )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    complete_count = sum(row["deep_model_complete"] == "true" for row in rows)
    lines = [
        "# C Deep Individual Model Status",
        "",
        f"- complete: {complete_count}/67",
        f"- remaining: {67 - complete_count}",
        "- completion requires a rule report, a structurally different builder, a candidate ONNX, and a cost-diff record",
        "- audit-only, Identity, opset-only, and initializer-only changes do not qualify",
        "",
        "| task | priority | baseline cost | complete | modeling | cost diff | builders | candidates |",
        "| --- | --- | ---: | --- | --- | --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['priority']} | {row['baseline_cost']} | "
            f"{row['deep_model_complete']} | {row['modeling_report']} | {row['cost_diff']} | "
            f"{row['builder_count']} | {row['candidate_count']} |"
        )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"complete": complete_count, "remaining": 67 - complete_count})


if __name__ == "__main__":
    main()
