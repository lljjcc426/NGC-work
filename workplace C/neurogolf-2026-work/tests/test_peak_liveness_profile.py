from __future__ import annotations

from pathlib import Path

import onnx
from onnx import TensorProto, helper

from peak_liveness_profile import profile_model


def _write_branch_model(path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4])
    nodes = [
        helper.make_node("Abs", ["x"], ["left"]),
        helper.make_node("Neg", ["x"], ["right"]),
        helper.make_node("Add", ["left", "right"], ["y"]),
    ]
    graph = helper.make_graph(nodes, "branch", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.save(model, path)


def test_profile_tracks_simultaneously_live_branches(tmp_path: Path) -> None:
    path = tmp_path / "branch.onnx"
    _write_branch_model(path)
    profile = profile_model(path)
    assert profile["activation_peak_bytes"] == 48
    assert profile["peak_node_index"] == 1
    assert profile["peak_node"]["op_type"] == "Neg"
    assert profile["nodes"][2]["released_after"] == ["left", "right"]
