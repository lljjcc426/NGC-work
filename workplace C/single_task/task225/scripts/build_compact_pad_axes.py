from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper

def build(source: Path, output: Path) -> Path:
    model=deepcopy(onnx.load(source));model.opset_import[0].version=18
    for name,values in {"axis2":[2],"axis3":[3],"axes_hw":[2,3],"pad_hw_compact":[0,0,24,24]}.items():
        model.graph.initializer.append(numpy_helper.from_array(np.array(values,dtype=np.int64),name=name))
    for node in model.graph.node:
        if node.op_type=="ReduceMin":
            axes=next(a for a in node.attribute if a.name=="axes");axis=int(axes.ints[0]);kept=[a for a in node.attribute if a.name!="axes"]
            del node.attribute[:];node.attribute.extend(kept);node.input.append("axis2" if axis==2 else "axis3")
        elif node.op_type=="Pad":
            node.input[1]="pad_hw_compact"
            while len(node.input)<3:node.input.append("")
            node.input.append("axes_hw")
    kept=[x for x in model.graph.initializer if x.name!="pad_hw"];del model.graph.initializer[:];model.graph.initializer.extend(kept)
    output.parent.mkdir(parents=True,exist_ok=True);onnx.checker.check_model(model,full_check=True);onnx.save(model,output);return output
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();print(build(a.source,a.output))
if __name__=="__main__":main()
