from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[2]
SINGLE = REPO / "workplace C" / "single_task"
SOURCE = Path(r"E:/kagglegolf/submissions/downloaded_best/v93_7273_37_user_upload/onnx")
sys.path.insert(0, str(SCRIPT_DIR))
from c_score_common import score_onnx  # noqa: E402


METHODS = {
    "task383": "crop_then_color_decode",
    "task382": "shared_color_projection_map",
    "task165": "explicit_float_template_correlation",
    "task378": "broadcast_coordinate_moments",
    "task132": "direct_grid_axis_reductions",
    "task069": "explicit_float_dynamic_correlations",
    "task284": "shared_marker_projection_map",
    "task201": "shared_frame_projection_map",
    "task224": "shared_gray_projection_map",
    "task094": "slice_reduce_line_detector",
}


def init(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(value, name=name)


def replace_at(model: onnx.ModelProto, index: int, nodes: list[onnx.NodeProto]) -> None:
    graph_nodes = list(model.graph.node)
    graph_nodes[index:index + 1] = nodes
    del model.graph.node[:]
    model.graph.node.extend(graph_nodes)


def add_inits(model: onnx.ModelProto, values: list[onnx.TensorProto]) -> None:
    model.graph.initializer.extend(values)


def shared_projection(model: onnx.ModelProto, first: int, second: int, selector: str, prefix: str) -> None:
    arrays = {x.name: numpy_helper.to_array(x) for x in model.graph.initializer}
    sel = arrays[selector].astype(np.float32).reshape(1, -1, 1, 1)
    sel_name = f"{prefix}_selector4"
    row_axes = f"{prefix}_row_axes"
    col_axes = f"{prefix}_col_axes"
    add_inits(model, [
        init(sel_name, sel),
        init(row_axes, np.array([1, 3], dtype=np.int64)),
        init(col_axes, np.array([1, 2], dtype=np.int64)),
    ])
    row_out = model.graph.node[first].output[0]
    col_out = model.graph.node[second].output[0]
    map_out = f"{prefix}_selected_map"
    nodes = list(model.graph.node)
    nodes[first:first + 1] = [
        helper.make_node("Mul", ["input", sel_name], [map_out], name=f"{prefix}_project"),
        helper.make_node("ReduceSum", [map_out, row_axes], [row_out], keepdims=0, name=f"{prefix}_row_reduce"),
    ]
    second += 1
    nodes[second:second + 1] = [
        helper.make_node("ReduceSum", [map_out, col_axes], [col_out], keepdims=0, name=f"{prefix}_col_reduce"),
    ]
    del model.graph.node[:]
    model.graph.node.extend(nodes)


def rewrite_task383(model: onnx.ModelProto) -> None:
    arrays = {x.name: numpy_helper.to_array(x) for x in model.graph.initializer}
    weights = arrays["cw"][0, :, 0, 0].astype(np.float32)
    add_inits(model, [
        init("deep_crop_starts", np.array([0, 0], dtype=np.int64)),
        init("deep_crop_ends", np.array([24, 24], dtype=np.int64)),
        init("deep_crop_axes", np.array([2, 3], dtype=np.int64)),
        init("deep_crop_steps", np.array([1, 1], dtype=np.int64)),
        init("deep_color_weights", weights),
        init("deep_unsqueeze_axis", np.array([1], dtype=np.int64)),
    ])
    replace_at(model, 0, [
        helper.make_node("Slice", ["input", "deep_crop_starts", "deep_crop_ends", "deep_crop_axes", "deep_crop_steps"], ["deep_crop"], name="deep_crop_24"),
        helper.make_node("Einsum", ["deep_crop", "deep_color_weights"], ["deep_color_hw"], equation="nchw,c->nhw", name="deep_color_decode"),
        helper.make_node("Unsqueeze", ["deep_color_hw", "deep_unsqueeze_axis"], ["deep_color_4d"], name="deep_color_channel"),
        helper.make_node("Add", ["deep_color_4d", "cb"], ["color_f"], name="deep_color_bias"),
    ])


def rewrite_task382(model: onnx.ModelProto) -> None:
    arrays = {x.name: numpy_helper.to_array(x) for x in model.graph.initializer}
    weights = arrays["color_selector"].astype(np.float32).reshape(3, 10, 1, 1)
    add_inits(model, [init("deep_color_conv_w", weights)])
    row_out = model.graph.node[0].output[0]
    col_out = model.graph.node[1].output[0]
    nodes = list(model.graph.node)
    nodes[0:2] = [
        helper.make_node("Conv", ["input", "deep_color_conv_w"], ["deep_color_map"], kernel_shape=[1, 1], name="deep_color_projection"),
        helper.make_node("ReduceSum", ["deep_color_map"], [row_out], axes=[3], keepdims=1, name="deep_row_projection"),
        helper.make_node("ReduceSum", ["deep_color_map"], [col_out], axes=[2], keepdims=1, name="deep_col_projection"),
    ]
    del model.graph.node[:]
    model.graph.node.extend(nodes)


def rewrite_task165(model: onnx.ModelProto) -> None:
    replace_at(model, 3, [
        helper.make_node("Cast", ["colored_u8"], ["deep_colored_f"], to=TensorProto.FLOAT, name="deep_colored_float"),
        helper.make_node("Cast", ["tmpl"], ["deep_tmpl_f"], to=TensorProto.FLOAT, name="deep_template_float"),
        helper.make_node("Conv", ["deep_colored_f", "deep_tmpl_f"], ["deep_kcount_f"], name="deep_template_correlation"),
        helper.make_node("Cast", ["deep_kcount_f"], ["kcount"], to=TensorProto.UINT8, name="deep_kcount_uint8"),
    ])


def rewrite_task378(model: onnx.ModelProto) -> None:
    w = np.arange(30, dtype=np.float32)
    add_inits(model, [
        init("deep_row_weights", w.reshape(1, 1, 30, 1)),
        init("deep_col_weights", w.reshape(1, 1, 1, 30)),
    ])
    nodes = list(model.graph.node)
    nodes[3:5] = [
        helper.make_node("Mul", ["input", "deep_row_weights"], ["deep_row_weighted"], name="deep_weight_rows"),
        helper.make_node("ReduceSum", ["deep_row_weighted"], ["rowmom"], axes=[2, 3], keepdims=0, name="deep_row_moment"),
        helper.make_node("Mul", ["input", "deep_col_weights"], ["deep_col_weighted"], name="deep_weight_cols"),
        helper.make_node("ReduceSum", ["deep_col_weighted"], ["colmom"], axes=[2, 3], keepdims=0, name="deep_col_moment"),
    ]
    del model.graph.node[:]
    model.graph.node.extend(nodes)


def rewrite_task132(model: onnx.ModelProto) -> None:
    add_inits(model, [
        init("deep_grid_row_axes", np.array([1, 3], dtype=np.int64)),
        init("deep_grid_col_axes", np.array([1, 2], dtype=np.int64)),
        init("deep_grid_start", np.array([0], dtype=np.int64)),
        init("deep_grid_end", np.array([15], dtype=np.int64)),
        init("deep_grid_axis", np.array([1], dtype=np.int64)),
    ])
    nodes = list(model.graph.node)
    nodes[25:26] = [
        helper.make_node("ReduceSum", ["input", "deep_grid_row_axes"], ["deep_grid_rows30"], keepdims=0, name="deep_grid_row_reduce"),
        helper.make_node("Slice", ["deep_grid_rows30", "deep_grid_start", "deep_grid_end", "deep_grid_axis"], ["gr_f"], name="deep_grid_rows15"),
    ]
    nodes[29:30] = [
        helper.make_node("ReduceSum", ["input", "deep_grid_col_axes"], ["deep_grid_cols30"], keepdims=0, name="deep_grid_col_reduce"),
        helper.make_node("Slice", ["deep_grid_cols30", "deep_grid_start", "deep_grid_end", "deep_grid_axis"], ["gc_f"], name="deep_grid_cols15"),
    ]
    del model.graph.node[:]
    model.graph.node.extend(nodes)


def qlinear_to_float(nodes: list[onnx.NodeProto], index: int, prefix: str) -> int:
    node = nodes[index]
    attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    x, weight = node.input[0], node.input[3]
    out = node.output[0]
    replacement = [
        helper.make_node("Cast", [x], [f"{prefix}_x_f"], to=TensorProto.FLOAT, name=f"{prefix}_x_float"),
        helper.make_node("Cast", [weight], [f"{prefix}_w_f"], to=TensorProto.FLOAT, name=f"{prefix}_w_float"),
        helper.make_node("Conv", [f"{prefix}_x_f", f"{prefix}_w_f"], [f"{prefix}_out_f"], name=f"{prefix}_conv", **attrs),
        helper.make_node("Cast", [f"{prefix}_out_f"], [out], to=TensorProto.UINT8, name=f"{prefix}_uint8"),
    ]
    nodes[index:index + 1] = replacement
    return len(replacement) - 1


def rewrite_task069(model: onnx.ModelProto) -> None:
    nodes = list(model.graph.node)
    offset = qlinear_to_float(nodes, 22, "deep_corr")
    qlinear_to_float(nodes, 28 + offset, "deep_place")
    del model.graph.node[:]
    model.graph.node.extend(nodes)


def rewrite_task284(model: onnx.ModelProto) -> None:
    shared_projection(model, 0, 1, "cmask", "deep_marker")


def rewrite_task201(model: onnx.ModelProto) -> None:
    shared_projection(model, 2, 3, "chan4", "deep_frame")


def rewrite_task224(model: onnx.ModelProto) -> None:
    shared_projection(model, 0, 12, "graysel", "deep_gray")


def rewrite_task094(model: onnx.ModelProto) -> None:
    add_inits(model, [
        init("deep_r0", np.array([0], dtype=np.int64)), init("deep_r2", np.array([2], dtype=np.int64)),
        init("deep_r4", np.array([4], dtype=np.int64)), init("deep_r9", np.array([9], dtype=np.int64)),
        init("deep_r11", np.array([11], dtype=np.int64)), init("deep_r13", np.array([13], dtype=np.int64)),
        init("deep_axis_h", np.array([2], dtype=np.int64)), init("deep_axis_w", np.array([3], dtype=np.int64)),
        init("deep_reduce_h", np.array([2], dtype=np.int64)), init("deep_reduce_w", np.array([3], dtype=np.int64)),
        init("deep_two", np.array(2.0, dtype=np.float32)),
    ])
    arrays = {x.name: x for x in model.graph.initializer}
    arrays["thr"].CopyFrom(init("thr", np.array([20.5], dtype=np.float32)))
    row = [
        helper.make_node("Slice", ["blue_f", "deep_r0", "deep_r9", "deep_axis_h"], ["deep_rt"], name="deep_row_top"),
        helper.make_node("Slice", ["blue_f", "deep_r2", "deep_r11", "deep_axis_h"], ["deep_rm"], name="deep_row_mid"),
        helper.make_node("Slice", ["blue_f", "deep_r4", "deep_r13", "deep_axis_h"], ["deep_rb"], name="deep_row_bottom"),
        helper.make_node("ReduceSum", ["deep_rt", "deep_reduce_w"], ["deep_rts"], keepdims=1, name="deep_row_top_sum"),
        helper.make_node("ReduceSum", ["deep_rm", "deep_reduce_w"], ["deep_rms"], keepdims=1, name="deep_row_mid_sum"),
        helper.make_node("ReduceSum", ["deep_rb", "deep_reduce_w"], ["deep_rbs"], keepdims=1, name="deep_row_bottom_sum"),
        helper.make_node("Add", ["deep_rts", "deep_rbs"], ["deep_ro"], name="deep_row_outer"),
        helper.make_node("Mul", ["deep_ro", "deep_two"], ["deep_ro2"], name="deep_row_outer_weight"),
        helper.make_node("Add", ["deep_ro2", "deep_rms"], ["row_score"], name="deep_row_score"),
    ]
    col = [
        helper.make_node("Slice", ["blue_f", "deep_r0", "deep_r9", "deep_axis_w"], ["deep_cl"], name="deep_col_left"),
        helper.make_node("Slice", ["blue_f", "deep_r2", "deep_r11", "deep_axis_w"], ["deep_cm"], name="deep_col_mid"),
        helper.make_node("Slice", ["blue_f", "deep_r4", "deep_r13", "deep_axis_w"], ["deep_cr"], name="deep_col_right"),
        helper.make_node("ReduceSum", ["deep_cl", "deep_reduce_h"], ["deep_cls"], keepdims=1, name="deep_col_left_sum"),
        helper.make_node("ReduceSum", ["deep_cm", "deep_reduce_h"], ["deep_cms"], keepdims=1, name="deep_col_mid_sum"),
        helper.make_node("ReduceSum", ["deep_cr", "deep_reduce_h"], ["deep_crs"], keepdims=1, name="deep_col_right_sum"),
        helper.make_node("Add", ["deep_cls", "deep_crs"], ["deep_co"], name="deep_col_outer"),
        helper.make_node("Mul", ["deep_co", "deep_two"], ["deep_co2"], name="deep_col_outer_weight"),
        helper.make_node("Add", ["deep_co2", "deep_cms"], ["col_score"], name="deep_col_score"),
    ]
    nodes = list(model.graph.node)
    nodes[2:4] = row + col
    del model.graph.node[:]
    model.graph.node.extend(nodes)


REWRITERS = {
    "task383": rewrite_task383, "task382": rewrite_task382, "task165": rewrite_task165,
    "task378": rewrite_task378, "task132": rewrite_task132, "task069": rewrite_task069,
    "task284": rewrite_task284, "task201": rewrite_task201, "task224": rewrite_task224,
    "task094": rewrite_task094,
}


def write_reports(task: str, old, new, output: Path) -> None:
    reports = SINGLE / task / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    delta = None if old.cost is None or new.cost is None else new.cost - old.cost
    accepted = bool(new.ok and delta is not None and delta < 0)
    row = {
        "task": task, "method": METHODS[task], "old_cost": old.cost, "new_cost": new.cost,
        "delta_cost": delta, "old_points": old.points, "new_points": new.points,
        "delta_points": None if old.points is None or new.points is None else new.points - old.points,
        "examples_passed": new.examples_passed, "examples_checked": new.examples_checked,
        "local_valid": str(new.ok).lower(), "accepted": str(accepted).lower(), "artifact_path": str(output),
    }
    with (reports / "cost_diff.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader(); writer.writerow(row)
    rule = METHODS[task].replace("_", " ")
    (reports / "modeling.md").write_text(
        f"# {task} Deep Model\n\n- rule implementation: {rule}\n- source: exact v93 task graph\n"
        f"- independent structure: yes; this changes the computation graph, not metadata or padding constants\n"
        f"- validation: {new.examples_passed}/{new.examples_checked}\n- old cost: {old.cost}\n- new cost: {new.cost}\n"
        f"- accepted: {str(accepted).lower()}\n- candidate: `{output}`\n\n"
        "The candidate is retained as a measured structural hypothesis. It is merged only when all examples pass and cost decreases.\n",
        encoding="utf-8",
    )
    (reports / f"FINAL_{task.upper()}_REPORT.md").write_text(
        f"# Final {task}\n\n- method: {rule}\n- full validation: {new.examples_passed}/{new.examples_checked}\n"
        f"- old cost: {old.cost}\n- new cost: {new.cost}\n- delta cost: {delta}\n- accepted: {str(accepted).lower()}\n"
        f"- artifact: `{output}`\n- next action: {'graft into current best' if accepted else 'retain as a falsified cost hypothesis and redesign'}\n",
        encoding="utf-8",
    )


def build(task: str) -> dict:
    if task not in REWRITERS:
        raise ValueError(task)
    source = SOURCE / f"{task}.onnx"
    output = SINGLE / task / "onnx" / f"{task}_{METHODS[task]}.onnx"
    output.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(source)
    REWRITERS[task](model)
    onnx.checker.check_model(model)
    onnx.save(model, output)
    old = score_onnx(task, source, validate_all=True)
    new = score_onnx(task, output, validate_all=True)
    write_reports(task, old, new, output)
    result = {"task": task, "method": METHODS[task], "old": asdict(old), "new": asdict(new)}
    print(json.dumps(result, indent=2))
    return result
