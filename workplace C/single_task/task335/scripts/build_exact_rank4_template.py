from __future__ import annotations

import csv
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


TASK = "task335"
HERE = Path(__file__).resolve()
TASK_DIR = HERE.parents[1]
REPO = HERE.parents[4]
COMMON = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"
BASE = Path(r"E:/kagglegolf/submissions/downloaded_best/v93_7273_37_user_upload/onnx/task335.onnx")
OUT = TASK_DIR / "onnx" / "task335_exact_rank4_template.onnx"


def exact_row_factor(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    basis_rows: list[np.ndarray] = []
    for row in matrix:
        if not basis_rows:
            if np.any(row):
                basis_rows.append(row)
            continue
        old_rank = np.linalg.matrix_rank(np.stack(basis_rows).astype(np.float64))
        trial = np.stack([*basis_rows, row]).astype(np.float64)
        if np.linalg.matrix_rank(trial) > old_rank:
            basis_rows.append(row)
        if len(basis_rows) == 4:
            break
    basis = np.stack(basis_rows).astype(np.float64)
    coefficients = []
    for row in matrix.astype(np.float64):
        solution = np.linalg.lstsq(basis.T, row, rcond=None)[0]
        rounded = np.rint(solution)
        if not np.array_equal(rounded @ basis, row):
            raise RuntimeError("task335 factorization is not exact over integers")
        coefficients.append(rounded)
    return np.stack(coefficients).astype(np.float16), basis.astype(np.float16)


def build_onnx(path: Path = OUT) -> Path:
    model = deepcopy(onnx.load(BASE))
    tensor = next(item for item in model.graph.initializer if item.name == "T")
    original = numpy_helper.to_array(tensor)
    coeff, basis = exact_row_factor(original.reshape(10, 16))
    left = coeff.reshape(1, 10, 4)
    right = basis.reshape(4, 4, 4)
    if not np.array_equal(np.einsum("bkp,pij->bkij", left, right), original):
        raise RuntimeError("rank-4 factors do not reconstruct T")

    target = model.graph.node[32]
    if target.op_type != "Einsum" or len(target.input) != 3 or target.input[0] != "T":
        raise RuntimeError(f"unexpected task335 terminal Einsum: {list(target.input)}")
    target.input[0] = "T_channel_factor"
    target.input.insert(1, "T_spatial_factor")
    for attribute in target.attribute:
        if attribute.name == "equation":
            attribute.s = b"bkp,pij,ir,jc->bkrc"

    kept = [item for item in model.graph.initializer if item.name != "T"]
    kept.extend([
        numpy_helper.from_array(left, name="T_channel_factor"),
        numpy_helper.from_array(right, name="T_spatial_factor"),
    ])
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    del model.graph.value_info[:]
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, path)
    return path


def main() -> None:
    sys.path.insert(0, str(COMMON))
    from c_score_common import score_onnx

    candidate = build_onnx()
    old = score_onnx(TASK, BASE, True)
    new = score_onnx(TASK, candidate, True)
    row = {
        "task": TASK, "method": "exact_rank4_template", "old_cost": old.cost,
        "new_cost": new.cost, "delta_cost": None if old.cost is None or new.cost is None else new.cost - old.cost,
        "old_points": old.points, "new_points": new.points,
        "delta_points": None if old.points is None or new.points is None else new.points - old.points,
        "examples_passed": new.examples_passed, "examples_checked": new.examples_checked,
        "local_valid": str(new.ok).lower(), "accepted": str(bool(new.ok and new.cost < old.cost)).lower(),
        "artifact_path": str(candidate),
    }
    report = TASK_DIR / "reports" / "cost_diff_round2.csv"
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
    print(row)


if __name__ == "__main__":
    main()
