from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper

def build(source: Path, output: Path) -> Path:
    model=deepcopy(onnx.load(source));model.opset_import[0].version=18
    for name,values in {"axes_013":[0,1,3],"axis0":[0],"axes_hw":[2,3],"pad_patch_compact":[0,0,2,3],"pad_out_compact":[0,0,20,20]}.items():
        model.graph.initializer.append(numpy_helper.from_array(np.array(values,dtype=np.int64),name=name))
    for node in model.graph.node:
        if node.op_type=="ReduceMax":
            axes=next(a for a in node.attribute if a.name=="axes");values=list(axes.ints)
            kept=[a for a in node.attribute if a.name!="axes"];del node.attribute[:];node.attribute.extend(kept)
            node.input.append("axes_013" if values==[0,1,3] else ("r012" if values==[0,1,2] else "r0123"))
        elif node.op_type=="Unsqueeze":
            del node.attribute[:];node.input.append("axis0")
        elif node.op_type=="Pad":
            old=node.input[1];node.input[1]="pad_patch_compact" if node.output[0]=="Kpad" else "pad_out_compact";node.input.append("axes_hw")
    kept=[x for x in model.graph.initializer if x.name not in {"pad_patch","pad_out"}];del model.graph.initializer[:];model.graph.initializer.extend(kept)
    output.parent.mkdir(parents=True,exist_ok=True);onnx.checker.check_model(model,full_check=True);onnx.save(model,output);return output
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();print(build(a.source,a.output))
if __name__=="__main__":main()
