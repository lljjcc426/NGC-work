from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper

def build(source: Path, output: Path) -> Path:
    model=deepcopy(onnx.load(source));model.opset_import[0].version=18
    for name,values in {"axes_hw":[2,3],"axis2":[2],"pad7_compact":[0,0,1,1],"pad30_compact":[0,0,18,18]}.items():
        model.graph.initializer.append(numpy_helper.from_array(np.array(values,dtype=np.int64),name=name))
    for node in model.graph.node:
        if node.op_type in {"ReduceSum","ReduceMax"}:
            axes=next(a for a in node.attribute if a.name=="axes");values=list(axes.ints)
            kept=[a for a in node.attribute if a.name!="axes"];del node.attribute[:];node.attribute.extend(kept)
            node.input.append("axes_hw" if values==[2,3] else ("ax1" if values==[1] else "axis2"))
        elif node.op_type=="Pad":
            old=node.input[1];node.input[1]="pad7_compact" if node.output[0]=="base7" else "pad30_compact";node.input.append("axes_hw")
    kept=[x for x in model.graph.initializer if x.name not in {"pad7","pad30"}];del model.graph.initializer[:];model.graph.initializer.extend(kept)
    output.parent.mkdir(parents=True,exist_ok=True);onnx.checker.check_model(model,full_check=True);onnx.save(model,output);return output
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();print(build(a.source,a.output))
if __name__=="__main__":main()
