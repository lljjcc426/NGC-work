from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper

def build(source: Path, output: Path) -> Path:
    model=deepcopy(onnx.load(source));model.opset_import[0].version=18
    model.graph.initializer.extend([numpy_helper.from_array(np.array([0,0,20,20],dtype=np.int64),name="pads_compact"),numpy_helper.from_array(np.array([2,3],dtype=np.int64),name="pad_axes_hw")])
    for node in model.graph.node:
        if node.op_type=="Pad":
            node.input[1]="pads_compact"
            while len(node.input)<3: node.input.append("")
            node.input.append("pad_axes_hw")
    kept=[x for x in model.graph.initializer if x.name!="pads"];del model.graph.initializer[:];model.graph.initializer.extend(kept)
    output.parent.mkdir(parents=True,exist_ok=True);onnx.checker.check_model(model,full_check=True);onnx.save(model,output);return output
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();print(build(a.source,a.output))
if __name__=="__main__":main()
