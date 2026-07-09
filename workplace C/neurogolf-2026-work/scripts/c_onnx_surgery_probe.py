from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import onnx

from c_score_common import (
    ARTIFACTS_DIR,
    CURRENT_BEST_ONNX_DIR,
    SCORE_DOCS,
    ensure_dirs,
    p0_p1_tasks,
    rel_to_kagglegolf,
    score_onnx,
    score_result_row,
    task_manifest,
    write_csv,
    write_md,
)


OPT_PASSES = [
    "eliminate_deadend",
    "eliminate_identity",
    "eliminate_nop_cast",
    "eliminate_nop_dropout",
    "eliminate_nop_flatten",
    "eliminate_nop_monotone_argmax",
    "eliminate_nop_pad",
    "eliminate_nop_transpose",
    "eliminate_unused_initializer",
    "extract_constant_to_initializer",
    "fuse_add_bias_into_conv",
    "fuse_consecutive_concats",
    "fuse_consecutive_log_softmax",
    "fuse_consecutive_reduce_unsqueeze",
    "fuse_consecutive_squeezes",
    "fuse_consecutive_transposes",
    "fuse_matmul_add_bias_into_gemm",
    "fuse_pad_into_conv",
    "fuse_transpose_into_gemm",
]


def parse_tasks(value: str) -> list[str]:
    if value.upper() == "P0P1":
        return p0_p1_tasks()
    if value.upper() == "ALLC":
        return [r["task"] for r in task_manifest()]
    return [x.strip() for x in value.split(",") if x.strip()]


def available_optimizer_passes() -> list[str]:
    try:
        import onnxoptimizer

        names = set(onnxoptimizer.get_available_passes())
        return [p for p in OPT_PASSES if p in names]
    except Exception:
        return []


def optimizer_pass(in_path: Path, out_path: Path) -> str:
    import onnxoptimizer

    passes = available_optimizer_passes()
    model = onnx.load(str(in_path))
    optimized = onnxoptimizer.optimize(model, passes)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(optimized, str(out_path))
    return ",".join(passes)


def sim_pass(in_path: Path, out_path: Path) -> str:
    from onnxsim import simplify

    model = onnx.load(str(in_path))
    simplified, ok = simplify(model, input_shapes={"input": [1, 10, 30, 30]})
    if not ok:
        raise RuntimeError("onnxsim check failed")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(simplified, str(out_path))
    return "onnxsim(input=1x10x30x30)"


def count_graph(path: Path) -> dict[str, int]:
    model = onnx.load(str(path))
    return {
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "value_info": len(model.graph.value_info),
        "file_size": path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="P0P1")
    parser.add_argument("--strategies", default="optimizer,sim,optimizer_sim")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--full-validate", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    tasks = parse_tasks(args.tasks)
    strategies = [x.strip() for x in args.strategies.split(",") if x.strip()]
    out_root = ARTIFACTS_DIR / "surgery_probes" / datetime.now().strftime("%Y%m%d_%H%M%S")
    rows: list[dict] = []
    accepted: list[dict] = []

    for task in tasks:
        old_path = CURRENT_BEST_ONNX_DIR / f"{task}.onnx"
        if not old_path.exists():
            rows.append({"task": task, "strategy": "baseline", "ok": False, "error": f"missing {old_path}"})
            continue
        old_score = score_onnx(task, old_path, validate_all=args.full_validate, max_examples=args.max_examples)
        old_row = score_result_row(old_score, current_cost=old_score.cost, source_label=rel_to_kagglegolf(old_path))
        old_graph = count_graph(old_path)
        rows.append(
            {
                "task": task,
                "strategy": "baseline",
                "old_cost": old_score.cost,
                "new_cost": old_score.cost,
                "delta_cost": 0,
                "ok": old_score.ok,
                "examples_checked": old_score.examples_checked,
                "nodes_before": old_graph["nodes"],
                "nodes_after": old_graph["nodes"],
                "initializers_before": old_graph["initializers"],
                "initializers_after": old_graph["initializers"],
                "file_size_before": old_graph["file_size"],
                "file_size_after": old_graph["file_size"],
                "accepted": False,
                "artifact_path": str(old_path),
                "notes": old_row.get("error", ""),
            }
        )

        generated: dict[str, Path] = {}
        for strategy in strategies:
            out_path = out_root / strategy / f"{task}.onnx"
            try:
                if strategy == "optimizer":
                    notes = optimizer_pass(old_path, out_path)
                elif strategy == "sim":
                    notes = sim_pass(old_path, out_path)
                elif strategy == "optimizer_sim":
                    mid_path = out_root / "_tmp_optimizer" / f"{task}.onnx"
                    notes1 = optimizer_pass(old_path, mid_path)
                    notes2 = sim_pass(mid_path, out_path)
                    notes = f"{notes1};{notes2}"
                else:
                    raise ValueError(f"unknown strategy: {strategy}")
                generated[strategy] = out_path
                result = score_onnx(task, out_path, validate_all=args.full_validate, max_examples=args.max_examples)
                graph = count_graph(out_path)
                delta = (old_score.cost - result.cost) if old_score.cost is not None and result.cost is not None else ""
                is_accepted = bool(result.ok and old_score.cost is not None and result.cost is not None and result.cost < old_score.cost)
                row = {
                    "task": task,
                    "strategy": strategy,
                    "old_cost": old_score.cost,
                    "new_cost": result.cost,
                    "delta_cost": delta,
                    "old_points": old_score.points,
                    "new_points": result.points,
                    "ok": result.ok,
                    "examples_checked": result.examples_checked,
                    "nodes_before": old_graph["nodes"],
                    "nodes_after": graph["nodes"],
                    "initializers_before": old_graph["initializers"],
                    "initializers_after": graph["initializers"],
                    "file_size_before": old_graph["file_size"],
                    "file_size_after": graph["file_size"],
                    "accepted": is_accepted,
                    "artifact_path": str(out_path),
                    "notes": notes if not result.error else f"{notes}; {result.error}",
                }
            except Exception as exc:
                row = {
                    "task": task,
                    "strategy": strategy,
                    "old_cost": old_score.cost,
                    "new_cost": "",
                    "delta_cost": "",
                    "ok": False,
                    "accepted": False,
                    "artifact_path": str(out_path),
                    "notes": f"{type(exc).__name__}: {exc}",
                }
            rows.append(row)
            if row.get("accepted") is True:
                accepted.append(row)

    out_csv = SCORE_DOCS / "artifact_scans" / "c_surgery_probe_results.csv"
    write_csv(out_csv, rows)
    write_csv(out_root / "surgery_probe_results.csv", rows)
    if accepted:
        write_csv(ARTIFACTS_DIR / "accepted_surgery_improvements.csv", accepted)

    lines = [
        "# C ONNX Surgery Probe",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Tasks: `{', '.join(tasks)}`",
        f"Strategies: `{', '.join(strategies)}`",
        f"Output root: `{out_root}`",
        f"Accepted improvements: `{len(accepted)}`",
        "",
        "| task | strategy | old_cost | new_cost | delta_cost | nodes | initializers | file_size | accepted | notes |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        nodes = f"{row.get('nodes_before','')}->{row.get('nodes_after','')}"
        inits = f"{row.get('initializers_before','')}->{row.get('initializers_after','')}"
        sizes = f"{row.get('file_size_before','')}->{row.get('file_size_after','')}"
        lines.append(
            f"| {row.get('task','')} | {row.get('strategy','')} | {row.get('old_cost','')} | {row.get('new_cost','')} | "
            f"{row.get('delta_cost','')} | {nodes} | {inits} | {sizes} | {row.get('accepted','')} | "
            f"`{str(row.get('notes',''))[:120]}` |"
        )
    write_md(SCORE_DOCS / "28_ONNX_SURGERY_PROBE.md", "\n".join(lines))
    write_md(out_root / "SURGERY_PROBE.md", "\n".join(lines))
    print(SCORE_DOCS / "28_ONNX_SURGERY_PROBE.md")
    print(f"accepted={len(accepted)} rows={len(rows)} out={out_root}")


if __name__ == "__main__":
    main()
