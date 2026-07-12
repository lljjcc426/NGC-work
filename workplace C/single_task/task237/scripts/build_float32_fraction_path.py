from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper

def build(source: Path, output: Path) -> Path:
    model=deepcopy(onnx.load(source));kept=[]
    for node in model.graph.node:
        if node.op_type=="Cast" and node.output and node.output[0]=="packed_grid_h":continue
        for i,value in enumerate(node.input):
            if value=="packed_grid_h":node.input[i]="packed_grid"
        kept.append(node)
    del model.graph.node[:];model.graph.node.extend(kept)
    for init in model.graph.initializer:
        if init.name in {"one_h","col_index"}:
            init.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float32),name=init.name))
    del model.graph.value_info[:]
    output.parent.mkdir(parents=True,exist_ok=True);onnx.checker.check_model(model,full_check=True);onnx.save(model,output);return output
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();print(build(a.source,a.output))
if __name__=="__main__":main()
