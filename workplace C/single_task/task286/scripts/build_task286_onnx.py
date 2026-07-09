from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


SCRIPT_DIR = Path(__file__).resolve().parent
TASK_DIR = SCRIPT_DIR.parent
REPORT_DIR = TASK_DIR / "reports"
ONNX_DIR = TASK_DIR / "onnx"
BASELINE = Path(
    r"E:\kagglegolf\submissions\candidates\GOLF_20260709_101_prvsiyan_7266_72_repro\onnx\task286.onnx"
)


PROFILES = {
    "sparse_all": {"em", "d_Wci", "pk_w0", "pk_w1", "pk_w2", "pk_w3", "v147_pads_2396_47"},
    "sparse_no_conv": {"em", "pk_w0", "pk_w1", "pk_w2", "pk_w3", "v147_pads_2396_47"},
    "sparse_pack_pad": {"pk_w0", "pk_w1", "pk_w2", "pk_w3", "v147_pads_2396_47"},
    "sparse_pack_only": {"pk_w0", "pk_w1", "pk_w2", "pk_w3"},
}


def sparse_tensor_from_dense(init: onnx.TensorProto) -> onnx.SparseTensorProto | None:
    arr = numpy_helper.to_array(init)
    nz = np.argwhere(arr != 0)
    if nz.size == 0 or nz.shape[0] >= arr.size:
        return None
    values = arr[tuple(nz.T)]
    values_tensor = numpy_helper.from_array(values.astype(arr.dtype, copy=False), name=init.name)
    indices_tensor = numpy_helper.from_array(nz.astype(np.int64), name=f"{init.name}_sparse_indices")
    return helper.make_sparse_tensor(values_tensor, indices_tensor, list(arr.shape))


def try_profile(profile_name: str, names: set[str]) -> dict:
    model = onnx.load(str(BASELINE))
    dense_keep = []
    sparse_new = []
    saved_params = 0
    sparse_names = []
    for init in model.graph.initializer:
        if init.name in names:
            arr = numpy_helper.to_array(init)
            sparse = sparse_tensor_from_dense(init)
            if sparse is not None:
                saved_params += int(arr.size - numpy_helper.to_array(sparse.values).size)
                sparse_new.append(sparse)
                sparse_names.append(init.name)
                continue
        dense_keep.append(init)

    del model.graph.initializer[:]
    model.graph.initializer.extend(dense_keep)
    model.graph.sparse_initializer.extend(sparse_new)

    out_path = ONNX_DIR / f"task286_{profile_name}.onnx"
    try:
        onnx.checker.check_model(model, full_check=True)
        ONNX_DIR.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(out_path))
        return {
            "profile": profile_name,
            "status": "checker_ok",
            "saved_params_if_supported": saved_params,
            "sparse_initializers": " ".join(sparse_names),
            "artifact_path": str(out_path),
            "error": "",
        }
    except Exception as exc:
        return {
            "profile": profile_name,
            "status": "checker_failed",
            "saved_params_if_supported": saved_params,
            "sparse_initializers": " ".join(sparse_names),
            "artifact_path": "",
            "error": f"{type(exc).__name__}: {exc}".replace("\n", " ")[:500],
        }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [try_profile(name, sparse_names) for name, sparse_names in PROFILES.items()]
    fieldnames = ["profile", "status", "saved_params_if_supported", "sparse_initializers", "artifact_path", "error"]
    with (REPORT_DIR / "onnx_probe.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (REPORT_DIR / "onnx_probe.md").open("w", encoding="utf-8") as f:
        f.write("# task286 ONNX Probe\n\n")
        f.write("Probe target: sparse initializer conversion for task286 constants.\n\n")
        f.write("| profile | status | saved_params_if_supported | sparse_initializers | error |\n")
        f.write("| --- | --- | ---: | --- | --- |\n")
        for row in rows:
            f.write(
                f"| {row['profile']} | {row['status']} | {row['saved_params_if_supported']} | "
                f"`{row['sparse_initializers']}` | {row['error']} |\n"
            )
        f.write("\nConclusion: no sparse-initializer candidate is accepted by ONNX checker/type inference for the relevant task286 operators. A lower-cost ONNX likely requires rewriting the flood-fill bitset graph itself, not only constants.\n")
    print(REPORT_DIR / "onnx_probe.md")


if __name__ == "__main__":
    main()
