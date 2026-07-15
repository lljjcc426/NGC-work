from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


GRID_SHAPE = [1, 10, 30, 30]
PUBLIC_SOURCE = "https://github.com/MinseongS/neurogolf"
PUBLIC_COMMIT = "be211ac481b1ba318d100adcfd09556392293be9"


def _make_einsum(equation: str, inputs: list[str], initializers: list[onnx.TensorProto]) -> onnx.ModelProto:
    tensor = lambda name: helper.make_tensor_value_info(name, TensorProto.FLOAT, GRID_SHAPE)
    graph = helper.make_graph(
        [helper.make_node("Einsum", inputs, ["output"], name="output", equation=equation)],
        "self_einsum_candidate",
        [tensor("input")],
        [tensor("output")],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model


def build_task017() -> onnx.ModelProto:
    # The same foreground-channel gate is deliberately shared by k/u/v.
    foreground = np.asarray([0, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
    equation = "bkxy,bura,buxa,bvzc,bvzy,k,u,v->bkrc"
    return _make_einsum(
        equation,
        ["input", "input", "input", "input", "input", "foreground", "foreground", "foreground"],
        [numpy_helper.from_array(foreground, "foreground")],
    )


def build_task197() -> onnx.ModelProto:
    row1 = np.zeros(30, dtype=np.float32)
    row1[1] = 1
    mixer = np.eye(10, dtype=np.float32)
    mixer[0, 1:] = -10
    equation = "bvrx,buyc,buzx,y,z,kv->bkrc"
    return _make_einsum(
        equation,
        ["input", "input", "input", "row1", "row1", "mixer"],
        [numpy_helper.from_array(row1, "row1"), numpy_helper.from_array(mixer, "mixer")],
    )


BUILDERS = {"task017": build_task017, "task197": build_task197}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build exact low-parameter self-Einsum candidates.")
    parser.add_argument("--tasks", nargs="+", choices=sorted(BUILDERS), default=sorted(BUILDERS))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in args.tasks:
        output = args.output_dir / f"{task}.onnx"
        model = BUILDERS[task]()
        onnx.save(model, output)
        rows.append(
            {
                "task": task,
                "path": str(output),
                "nodes": len(model.graph.node),
                "params": sum(int(np.prod(init.dims)) for init in model.graph.initializer),
                "equation": helper.get_attribute_value(model.graph.node[0].attribute[0]).decode(),
                "source": PUBLIC_SOURCE,
                "source_commit": PUBLIC_COMMIT,
            }
        )
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
