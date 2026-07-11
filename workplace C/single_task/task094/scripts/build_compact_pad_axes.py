from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper

SPECS = {"blue15": [1, 1, 1, 1], "line_bg30": [0, 0, 15, 15]}

def build(source: Path, output: Path) -> Path:
    model = deepcopy(onnx.load(source)); model.opset_import[0].version = 18
    model.graph.initializer.append(numpy_helper.from_array(np.array([2, 3], dtype=np.int64), name="pad_axes_hw"))
    removed = set()
    for node in model.graph.node:
        if node.op_type == "Pad" and node.output[0] in SPECS:
            old = node.input[1]; new = old + "_compact"
            model.graph.initializer.append(numpy_helper.from_array(np.array(SPECS[node.output[0]], dtype=np.int64), name=new))
            node.input[1] = new
            while len(node.input) < 3: node.input.append("")
            node.input.append("pad_axes_hw"); removed.add(old)
    kept = [item for item in model.graph.initializer if item.name not in removed]
    del model.graph.initializer[:]; model.graph.initializer.extend(kept)
    output.parent.mkdir(parents=True, exist_ok=True); onnx.checker.check_model(model, full_check=True); onnx.save(model, output); return output

def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();print(build(a.source,a.output))
if __name__ == "__main__": main()
