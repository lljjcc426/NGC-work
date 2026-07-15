from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PARENT = (
    REPO
    / "workplace C"
    / "artifacts"
    / "full400_round37b_isolate_dilated_187"
    / "onnx"
    / "task110.onnx"
)
TASK_JSON = REPO / "neurogolf_400_tasks" / "tasks" / "task110.json"
SCORER_DIR = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"
MODEL_DIR = HERE / "onnx"
REPORT_DIR = HERE / "reports"
CANDIDATE = MODEL_DIR / "task110_cp65_exact.onnx"
REPORT = REPORT_DIR / "task110_cp65_exact_evidence.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initializer_map(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}


def set_equation(node: onnx.NodeProto, equation: str) -> None:
    for attr in node.attribute:
        if attr.name == "equation":
            attr.s = equation.encode("ascii")
            return
    node.attribute.append(helper.make_attribute("equation", equation))


def exact_cp_factors(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    if a.shape != (6, 30, 30):
        raise ValueError(f"unexpected A shape: {a.shape}")

    columns: list[np.ndarray] = []
    groups: list[dict[str, object]] = []

    start = 0
    for index in range(30):
        column = np.zeros(30, dtype=np.float32)
        column[index] = 1.0
        columns.append(column)
    groups.append({"slice": 0, "kind": "identity", "start": start, "stop": 30, "terms": 30})

    start = 30
    for slice_index, period in enumerate(range(5, 10), start=1):
        counts: list[int] = []
        for residue in range(period):
            column = np.zeros(30, dtype=np.float32)
            positions = [index for index in range(29) if index % period == residue]
            column[positions] = 1.0
            columns.append(column)
            counts.append(len(positions))
        stop = start + period
        groups.append(
            {
                "slice": slice_index,
                "kind": "residue_gram",
                "period": period,
                "start": start,
                "stop": stop,
                "terms": period,
                "permuted_block_sizes": counts + [1],
                "last_block_is_zero": True,
            }
        )
        start = stop

    b = np.stack(columns, axis=1)
    c = np.zeros((6, b.shape[1]), dtype=np.float32)
    for group in groups:
        c[int(group["slice"]), int(group["start"]):int(group["stop"])] = 1.0

    reconstructed = np.einsum("xk,pk,zk->pxz", b, c, b, optimize=True)
    if not np.array_equal(reconstructed, a):
        mismatch = int(np.count_nonzero(reconstructed != a))
        raise AssertionError(f"CP reconstruction mismatch: {mismatch}")
    return b, c, groups


def build_candidate(parent: onnx.ModelProto, b: np.ndarray, c: np.ndarray) -> onnx.ModelProto:
    model = onnx.ModelProto()
    model.CopyFrom(parent)
    graph = model.graph

    kept = [item for item in graph.initializer if item.name != "A"]
    del graph.initializer[:]
    graph.initializer.extend(kept)
    graph.initializer.extend(
        [
            numpy_helper.from_array(b.astype(np.float32), name="B"),
            numpy_helper.from_array(c.astype(np.float32), name="C"),
        ]
    )

    by_output = {node.output[0]: node for node in graph.node}
    row_node = by_output["confr"]
    del row_node.input[:]
    row_node.input.extend(["input", "input", "B", "B", "C", "D"])
    set_equation(row_node, "nvxy,nwzy,xk,zk,pk,vw->p")

    col_node = by_output["confc"]
    del col_node.input[:]
    col_node.input.extend(["input", "input", "B", "B", "C", "D"])
    set_equation(col_node, "nvxy,nwxz,yk,zk,pk,vw->p")

    output_node = by_output["output"]
    del output_node.input[:]
    output_node.input.extend(
        ["g_r", "g_c", "m", "input", "B", "B", "C", "B", "B", "C"]
    )
    set_equation(output_node, "p,q,v,nvxy,xk,rk,pk,yl,cl,ql->nvrc")

    model.graph.name = "task110_exact_shared_cp65"
    model.doc_string = "Exact shared CP factorization of the parent A tensor."
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model


def make_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        path.read_bytes(), options, providers=["CPUExecutionProvider"]
    )


def grid_to_tensor(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    array = np.asarray(grid, dtype=np.int64)
    result = np.zeros((1, 10, 30, 30), dtype=np.float32)
    rows, cols = array.shape
    rr, cc = np.indices((rows, cols))
    result[0, array, rr, cc] = 1.0
    return result


def detector_choice(x: np.ndarray, a: np.ndarray, d: np.ndarray, u: np.ndarray) -> tuple[int, int]:
    image = x[0]
    confr = np.einsum("vxy,wzy,pxz,vw->p", image, image, a, d, optimize=True)
    confc = np.einsum("vxy,wxz,pyz,vw->p", image, image, a, d, optimize=True)

    def choose(conflicts: np.ndarray) -> int:
        valid = conflicts == 0
        later_valid = np.einsum("q,qp->p", valid.astype(np.float32), u)
        selected = valid & (later_valid == 0)
        indices = np.flatnonzero(selected)
        return int(indices[0]) if len(indices) == 1 else -1

    return choose(confr), choose(confc)


def random_cases(
    a: np.ndarray,
    d: np.ndarray,
    u: np.ndarray,
    seed: int = 11039,
) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    cases: list[tuple[str, np.ndarray]] = []

    for row_slice in range(6):
        for col_slice in range(6):
            row_period = row_slice + 4 if row_slice else None
            col_period = col_slice + 4 if col_slice else None
            for attempt in range(2000):
                if row_period is None and col_period is None:
                    grid = rng.integers(1, 10, size=(29, 29), dtype=np.int64)
                elif row_period is None:
                    table = rng.integers(1, 10, size=(29, col_period), dtype=np.int64)
                    grid = table[:, np.arange(29) % col_period]
                elif col_period is None:
                    table = rng.integers(1, 10, size=(row_period, 29), dtype=np.int64)
                    grid = table[np.arange(29) % row_period, :]
                else:
                    table = rng.integers(
                        1, 10, size=(row_period, col_period), dtype=np.int64
                    )
                    grid = table[
                        (np.arange(29) % row_period)[:, None],
                        (np.arange(29) % col_period)[None, :],
                    ]
                tensor = grid_to_tensor(grid)
                if detector_choice(tensor, a, d, u) == (row_slice, col_slice):
                    cases.append(
                        (
                            f"slice_pair_{row_slice}_{col_slice}_attempt_{attempt}",
                            tensor,
                        )
                    )
                    break
            else:
                raise RuntimeError(
                    f"could not construct slice coverage case {(row_slice, col_slice)}"
                )

    for index in range(16):
        grid = rng.integers(0, 10, size=(29, 29), dtype=np.int64)
        cases.append((f"dense_{index}", grid_to_tensor(grid)))
    return cases


def compare_sessions(
    parent_session: ort.InferenceSession,
    candidate_session: ort.InferenceSession,
    cases: list[tuple[str, np.ndarray]],
    a: np.ndarray,
    d: np.ndarray,
    u: np.ndarray,
) -> dict[str, object]:
    mismatches: list[dict[str, object]] = []
    max_abs = 0.0
    coverage: Counter[str] = Counter()
    for name, x in cases:
        expected = parent_session.run(None, {"input": x})[0]
        actual = candidate_session.run(None, {"input": x})[0]
        difference = np.abs(expected - actual)
        case_max = float(difference.max(initial=0.0))
        max_abs = max(max_abs, case_max)
        choice = detector_choice(x, a, d, u)
        coverage[f"{choice[0]},{choice[1]}"] += 1
        if not np.array_equal(expected, actual):
            mismatches.append(
                {
                    "case": name,
                    "max_abs": case_max,
                    "elements": int(np.count_nonzero(expected != actual)),
                    "detector_choice": list(choice),
                }
            )
            if len(mismatches) >= 10:
                break
    return {
        "cases": len(cases),
        "passed": len(cases) - len(mismatches),
        "failed": len(mismatches),
        "array_equal_all": not mismatches,
        "max_abs_error": max_abs,
        "selected_slice_pair_coverage": dict(sorted(coverage.items())),
        "mismatches": mismatches,
    }


def official_raw_cases() -> list[tuple[str, np.ndarray]]:
    task = json.loads(TASK_JSON.read_text(encoding="utf-8"))
    cases: list[tuple[str, np.ndarray]] = []
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(task.get(split, [])):
            cases.append((f"{split}_{index}", grid_to_tensor(example["input"])))
    return cases


def main() -> None:
    if HERE != REPO / "workplace E" / "single_task" / "task110" / "agent_round39":
        raise RuntimeError(f"unexpected owner directory: {HERE}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    parent = onnx.load(PARENT)
    arrays = initializer_map(parent)
    a = arrays["A"].astype(np.float32)
    d = arrays["D"].astype(np.float32)
    u = arrays["U"].astype(np.float32)
    b, c, groups = exact_cp_factors(a)

    candidate = build_candidate(parent, b, c)
    onnx.save(candidate, CANDIDATE)

    sys.path.insert(0, str(SCORER_DIR))
    from c_score_common import score_onnx

    parent_score = score_onnx("task110", PARENT, validate_all=True)
    candidate_score = score_onnx("task110", CANDIDATE, validate_all=True)

    parent_session = make_session(PARENT)
    candidate_session = make_session(CANDIDATE)
    official_comparison = compare_sessions(
        parent_session,
        candidate_session,
        official_raw_cases()[:12],
        a,
        d,
        u,
    )
    randomized_comparison = compare_sessions(
        parent_session,
        candidate_session,
        random_cases(a, d, u),
        a,
        d,
        u,
    )

    commutator = a[1] @ a[2] - a[2] @ a[1]
    tensor_analysis = {
        "A_shape": list(a.shape),
        "A_elements": int(a.size),
        "A_nonzero": int(np.count_nonzero(a)),
        "exact_slice_ranks": [30, 5, 6, 7, 8, 9],
        "rank_basis": "I_30 and residue-class all-ones blocks after exact permutation",
        "exact_tucker_mode_ranks": [6, 30, 30],
        "tucker_rank_basis": "six independent slices; each spatial unfolding contains the I_30 slice",
        "cp": {
            "exact_constructive_terms": int(b.shape[1]),
            "exact_identity_verified": bool(
                np.array_equal(a, np.einsum("xk,pk,zk->pxz", b, c, b, optimize=True))
            ),
            "factor_shapes": {"B": list(b.shape), "C": list(c.shape)},
            "dense_factor_parameters": int(b.size + c.size),
            "rank_lower_bound": 31,
            "rank_upper_bound": int(b.shape[1]),
            "lower_bound_reason": "rank 30 would force simultaneous diagonalization because slice 0 is I_30, but slices 1 and 2 do not commute",
            "slice_1_2_commutator_nonzero": int(np.count_nonzero(commutator)),
        },
        "kronecker_block_analysis": groups,
        "kronecker_note": "For each period, residue permutation gives a direct sum of J blocks and one zero padding block; equal full blocks are J_k kron I_period before the 29-cell truncation.",
    }

    report = {
        "task": "task110",
        "owner_directory": str(HERE),
        "parent": {**asdict(parent_score), "expected_cost": 5811},
        "candidate": asdict(candidate_score),
        "cost_delta": (
            parent_score.cost - candidate_score.cost
            if parent_score.cost is not None and candidate_score.cost is not None
            else None
        ),
        "parameter_delta": (
            parent_score.params - candidate_score.params
            if parent_score.params is not None and candidate_score.params is not None
            else None
        ),
        "tensor_analysis": tensor_analysis,
        "official_parent_candidate_raw_sample_comparison": official_comparison,
        "random_parent_candidate_raw_comparison": randomized_comparison,
        "graph": {
            "parent_ops": dict(Counter(node.op_type for node in parent.graph.node)),
            "candidate_ops": dict(Counter(node.op_type for node in candidate.graph.node)),
            "parent_initializers": {key: list(value.shape) for key, value in arrays.items()},
            "candidate_initializers": {
                item.name: list(numpy_helper.to_array(item).shape)
                for item in candidate.graph.initializer
            },
        },
        "artifacts": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256(Path(__file__).resolve()),
            "model": str(CANDIDATE.resolve()),
            "model_sha256": sha256(CANDIDATE),
            "report": str(REPORT.resolve()),
        },
    }

    required = [
        parent_score.ok,
        candidate_score.ok,
        candidate_score.params is not None and candidate_score.params < parent_score.params,
        official_comparison["array_equal_all"],
        randomized_comparison["array_equal_all"],
        tensor_analysis["cp"]["exact_identity_verified"],
    ]
    report["all_required_checks_passed"] = bool(all(required))
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["all_required_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
