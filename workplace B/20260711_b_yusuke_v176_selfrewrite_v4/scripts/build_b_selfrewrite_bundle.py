from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import build_blend


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ROOT / "public_probe_variants"
ARTIFACTS = ROOT / "artifacts"


def parse_override(value: str) -> tuple[int, Path]:
    task_text, separator, path_text = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("override must be TASK=PATH")
    task = int(task_text.removeprefix("task"))
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    if not 1 <= task <= 400 or not path.is_file():
        raise argparse.ArgumentTypeError(f"invalid override: {value}")
    return task, path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-name", required=True)
    parser.add_argument("--out-name", required=True)
    parser.add_argument("--override", action="append", type=parse_override, default=[])
    args = parser.parse_args()

    base_root = VARIANTS / args.base_name
    base_dir = base_root / "submission"
    base_summary = json.loads((base_root / "summary.json").read_text(encoding="utf-8"))
    original_name = str(base_summary["base"])
    original_rows = json.loads(
        (ARTIFACTS / f"{original_name}_all_scores.json").read_text(encoding="utf-8")
    )
    original_by_task = {int(row["task"]): dict(row) for row in original_rows}
    if len(original_by_task) != 400:
        raise RuntimeError(f"original score table has {len(original_by_task)}/400 tasks")

    out_root = VARIANTS / args.out_name
    out_dir = out_root / "submission"
    if out_root.exists():
        shutil.rmtree(out_root)
    shutil.copytree(base_dir, out_dir)

    changed_by_task = {
        int(row["task"]): dict(row) for row in base_summary.get("changed_tasks", [])
    }
    for task, source in args.override:
        destination = out_dir / f"task{task:03d}.onnx"
        shutil.copy2(source, destination)
        scored = build_blend.validate_and_score((task, args.out_name, str(destination)))
        if not scored.get("valid") or scored.get("points") is None:
            raise RuntimeError(f"task{task:03d} failed validation: {scored.get('error')}")
        scored.update(
            base_cost=original_by_task[task]["cost"],
            base_points=original_by_task[task]["points"],
            gain=float(scored["points"]) - float(original_by_task[task]["points"]),
            override_source=str(source),
        )
        changed_by_task[task] = scored

    files = sorted(out_dir.glob("task*.onnx"))
    expected_names = {f"task{task:03d}.onnx" for task in range(1, 401)}
    if {path.name for path in files} != expected_names:
        raise RuntimeError("submission does not contain exactly task001..task400")

    zip_path = out_root / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=path.name)

    changed = [changed_by_task[task] for task in sorted(changed_by_task)]
    fields = [
        "task",
        "cost",
        "points",
        "base_cost",
        "base_points",
        "gain",
        "override_source",
    ]
    with (out_root / "changed_tasks.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(changed)

    original_total = sum(float(row["points"]) for row in original_rows)
    total_gain = sum(float(row["gain"]) for row in changed)
    summary = {
        "variant": args.out_name,
        "purpose": "B-only self-rewrite bundle over the online-safe baseline",
        "base_variant": args.base_name,
        "original_base": original_name,
        "original_local_total": original_total,
        "changed_count": len(changed),
        "combined_gain_vs_original": total_gain,
        "expected_local_total": original_total + total_gain,
        "changed_tasks": changed,
        "zip_path": str(zip_path.resolve()),
        "zip_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
        "zip_size": zip_path.stat().st_size,
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (ARTIFACTS / f"{args.out_name}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"built {args.out_name}: tasks={len(changed)} "
        f"gain={total_gain:+.6f} total={original_total + total_gain:.6f} "
        f"zip={zip_path}"
    )


if __name__ == "__main__":
    main()
