from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import onnx

def build(source: Path, output: Path) -> Path:
    model=deepcopy(onnx.load(source));changed=0
    for node in model.graph.node:
        if node.op_type=="QLinearMatMul":
            node.op_type="Mul";del node.input[:];node.input.extend(["row_mod","col_mod"]);changed+=1
    if changed!=1:raise RuntimeError(changed)
    kept=[x for x in model.graph.initializer if x.name not in {"q_scale","q_zero"}];del model.graph.initializer[:];model.graph.initializer.extend(kept)
    output.parent.mkdir(parents=True,exist_ok=True);onnx.checker.check_model(model,full_check=True);onnx.save(model,output);return output
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();print(build(a.source,a.output))
if __name__=="__main__":main()
