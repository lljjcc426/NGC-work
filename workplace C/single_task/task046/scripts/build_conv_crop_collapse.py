from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import onnx
from onnx import helper, numpy_helper

def build(source: Path, output: Path) -> Path:
    model=deepcopy(onnx.load(source));conv=model.graph.node[0]
    if conv.op_type!="Conv" or conv.input[1]!="color_weights":raise RuntimeError("unexpected task046 graph")
    weight=next(x for x in model.graph.initializer if x.name=="color_weights");compact=numpy_helper.to_array(weight)[:,:,:1,:1]
    weight.CopyFrom(numpy_helper.from_array(compact,name="color_weights"));del conv.attribute[:]
    conv.attribute.extend([helper.make_attribute("kernel_shape",[1,1]),helper.make_attribute("pads",[0,0,-27,-10])])
    output.parent.mkdir(parents=True,exist_ok=True);onnx.checker.check_model(model,full_check=True);onnx.save(model,output);return output
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();print(build(a.source,a.output))
if __name__=="__main__":main()
