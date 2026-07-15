from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = REPO_ROOT / "workplace C" / "neurogolf-2026-work" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from c_score_common import score_onnx  # noqa: E402


def sparse_constant(name: str, dims: list[int], coordinates: list[list[int]], values: list[float]) -> onnx.NodeProto:
    value_tensor = numpy_helper.from_array(np.asarray(values, dtype=np.float32), name=f"{name}_values")
    index_tensor = numpy_helper.from_array(np.asarray(coordinates, dtype=np.int64), name=f"{name}_indices")
    sparse = helper.make_sparse_tensor(value_tensor, index_tensor, dims)
    return helper.make_node("Constant", [], [name], name=name, sparse_value=sparse)


def build(output_path: Path) -> Path:
    # r = 3*a+p is the required Kronecker coordinate ordering. Only nine of
    # the 270 entries are non-zero.
    mapping_coordinates = []
    for a in range(3):
        for p in range(3):
            mapping_coordinates.append([3 * a + p, a, p])

    # The generator uses background 0 plus exactly one foreground color. For
    # every legal pair of cells, route the pair to background when either cell
    # is background, otherwise to that common foreground color.
    route_coordinates: list[list[int]] = []
    for d in range(10):
        route_coordinates.append([0, d, 0])
    for c in range(1, 10):
        route_coordinates.append([c, 0, 0])
        route_coordinates.append([c, c, c])

    nodes = [
        sparse_constant("route", [10, 10, 10], route_coordinates, [1.0] * len(route_coordinates)),
        sparse_constant("mapping", [30, 3, 3], mapping_coordinates, [1.0] * len(mapping_coordinates)),
        helper.make_node(
            "Einsum",
            ["input", "input", "route", "mapping", "mapping"],
            ["output"],
            name="task001_sparse_terminal",
            equation="ncab,ndpq,cde,rap,sbq->ners",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "task001_sparse_terminal_einsum",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
    )
    model = helper.make_model(
        graph,
        producer_name="ngc-task001-sparse-terminal-einsum",
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=10,
    )
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    reloaded = onnx.load(output_path)
    onnx.checker.check_model(reloaded, full_check=True)
    onnx.shape_inference.infer_shapes(reloaded, strict_mode=True)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "workplace B" / "single_task" / "task001" / "debug" / "task001_sparse_terminal_einsum.onnx",
    )
    args = parser.parse_args()
    path = build(args.output)
    result = score_onnx("task001", path)
    print(result)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
