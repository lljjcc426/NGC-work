from __future__ import annotations

from pathlib import Path

import onnx
import numpy as np
from onnx import TensorProto, helper
from onnx import numpy_helper


BASELINE = Path(
    r"E:/kagglegolf/submissions/candidates/"
    r"GOLF_20260709_101_prvsiyan_7266_72_repro/onnx/task054.onnx"
)
OUT_DIR = Path(r"E:/kongming/NGC-work/workplace C/single_task/task054/onnx")


def _remove_value_info(model: onnx.ModelProto, names: set[str]) -> None:
    for field in (model.graph.value_info, model.graph.output):
        keep = [item for item in field if item.name not in names]
        del field[:]
        field.extend(keep)


def build_variant(output_path: Path, reduction: str) -> Path:
    """Fuse the final line overlay into a scalar ScatterElements output.

    Baseline ending:
      line_mask = ScatterElements(... 0/1 updates) > rect_cleared
      final_scalar = Where(line_mask, line_color, neighbor_overlay)

    Candidate ending:
      existing = GatherElements(neighbor_overlay, line_indices)
      updates = Where(line_update != 0, line_color, existing)
      final_scalar = ScatterElements(neighbor_overlay, line_indices, updates)

    This keeps the upstream rule logic unchanged and only changes final scalar
    canvas composition.
    """
    model = onnx.load(str(BASELINE))

    # Remove the old line scatter, derived full-grid line mask, and final Where.
    removed_outputs = {"safe_name_318", "safe_name_319", "safe_name_338"}
    kept_nodes = [
        node
        for node in model.graph.node
        if not any(out in removed_outputs for out in node.output)
    ]

    new_nodes = [
        helper.make_node(
            "GatherElements",
            ["safe_name_337", "safe_name_317"],
            ["task054_line_existing"],
            axis=3,
        ),
        helper.make_node(
            "Equal",
            ["safe_name_313", "safe_name_3"],
            ["task054_line_cond"],
        ),
        helper.make_node(
            "Where",
            ["task054_line_cond", "safe_name_167", "task054_line_existing"],
            ["task054_line_updates"],
        ),
    ]
    attrs = {"axis": 3}
    if reduction:
        attrs["reduction"] = reduction
    new_nodes.append(
        helper.make_node(
            "ScatterElements",
            ["safe_name_337", "safe_name_317", "task054_line_updates"],
            ["safe_name_338"],
            **attrs,
        )
    )

    # Insert immediately before the final Equal node so dependencies are already
    # available and the original output contract remains unchanged.
    final_equal = kept_nodes[-1]
    if final_equal.op_type != "Equal" or list(final_equal.output) != ["output"]:
        raise RuntimeError("unexpected task054 graph ending")
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes[:-1] + new_nodes + [final_equal])

    _remove_value_info(model, removed_outputs | {n.output[0] for n in new_nodes})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))
    return output_path


def _replace_initializer(model: onnx.ModelProto, name: str, array: np.ndarray) -> None:
    keep = [init for init in model.graph.initializer if init.name != name]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    model.graph.initializer.append(numpy_helper.from_array(array, name=name))


def build_index32_variant(output_path: Path) -> Path:
    """Keep baseline output fusion, but lower the final 8-neighbor index path.

    The final local-template overlay uses a 4x8x4 index tensor for GatherND and
    ScatterND. ONNX accepts int32 indices here, while the baseline builds this
    subgraph in int64. This is a semantics-preserving memory cut.
    """
    model = onnx.load(str(BASELINE))

    _replace_initializer(model, "safe_name_26", np.zeros((4, 8, 1), dtype=np.int32))
    _replace_initializer(
        model,
        "safe_name_28",
        np.array([[-1, -1, -1, 0, 0, 1, 1, 1]], dtype=np.int32),
    )
    _replace_initializer(
        model,
        "safe_name_29",
        np.array([[-1, 0, 1, -1, 1, -1, 0, 1]], dtype=np.int32),
    )

    new_nodes = []
    for node in model.graph.node:
        if list(node.output) == ["safe_name_323"]:
            new_nodes.append(
                helper.make_node("Cast", ["safe_name_321"], ["task054_rows_i32"], to=TensorProto.INT32)
            )
            node.input[0] = "task054_rows_i32"
        elif list(node.output) == ["safe_name_324"]:
            new_nodes.append(
                helper.make_node("Cast", ["safe_name_322"], ["task054_cols_i32"], to=TensorProto.INT32)
            )
            node.input[0] = "task054_cols_i32"
        new_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    _remove_value_info(model, {"task054_rows_i32", "task054_cols_i32"})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))
    return output_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, reduction in [
        ("task054_fusion_scatter_none.onnx", ""),
        ("task054_fusion_scatter_max.onnx", "max"),
    ]:
        path = build_variant(OUT_DIR / name, reduction)
        print(path)
    print(build_index32_variant(OUT_DIR / "task054_index32_overlay.onnx"))


if __name__ == "__main__":
    main()
