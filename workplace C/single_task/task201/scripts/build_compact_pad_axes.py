from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper

def build(source: Path, output: Path) -> Path:
    model=deepcopy(onnx.load(source));model.opset_import[0].version=18
    for name,values in {
        "axes_13":[1,3],"axes_12":[1,2],"axes_hw":[2,3],
        "pads_middle_compact":[1,1,1,1],"pads_output_compact":[0,0,23,22],
    }.items(): model.graph.initializer.append(numpy_helper.from_array(np.array(values,dtype=np.int64),name=name))
    pad_specs={"pattern_full":("pads_middle_compact","pads_middle"),"output":("pads_output_compact","pads_output")}
    for node in model.graph.node:
        if node.op_type=="ReduceMax":
            axes=next(a for a in node.attribute if a.name=="axes"); values=list(axes.ints)
            kept=[a for a in node.attribute if a.name!="axes"];del node.attribute[:];node.attribute.extend(kept)
            if values==[1,3]:node.input.append("axes_13")
            elif values==[1,2]:node.input.append("axes_12")
            elif values!=[1,2,3]:raise RuntimeError(values)
        elif node.op_type=="Pad" and node.output[0] in pad_specs:
            new,_=pad_specs[node.output[0]];node.input[1]=new
            while len(node.input)<3:node.input.append("")
            node.input.append("axes_hw")
    removed={old for _,old in pad_specs.values()};kept=[x for x in model.graph.initializer if x.name not in removed]
    del model.graph.initializer[:];model.graph.initializer.extend(kept)
    output.parent.mkdir(parents=True,exist_ok=True);onnx.checker.check_model(model,full_check=True);onnx.save(model,output);return output
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();print(build(a.source,a.output))
if __name__=="__main__":main()
