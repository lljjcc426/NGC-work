from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import onnx

def build(source: Path, output: Path) -> Path:
    model=deepcopy(onnx.load(source));mapping={"rf_i32":"rf_i8","rl_i32":"rl_i8","cf_i32":"cf_i8","cl_i32":"cl_i8"};changed=0
    for node in model.graph.node:
        if node.op_type=="Cast" and node.output and node.output[0] in mapping:
            node.input[0]=mapping[node.output[0]];changed+=1
    if changed!=4:raise RuntimeError(changed)
    nodes=list(model.graph.node)
    for first in (6,8,10,12):
        nodes[first],nodes[first+1]=nodes[first+1],nodes[first]
    del model.graph.node[:];model.graph.node.extend(nodes)
    output.parent.mkdir(parents=True,exist_ok=True);onnx.checker.check_model(model,full_check=True);onnx.save(model,output);return output
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();print(build(a.source,a.output))
if __name__=="__main__":main()
