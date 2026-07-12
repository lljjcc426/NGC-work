from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import onnx
from onnx import numpy_helper

def build(source: Path, output: Path) -> Path:
    model=deepcopy(onnx.load(source))
    names={"q_scale","q_zero_u8","q_zero_i8","one_u8","y_scale5"}
    for init in model.graph.initializer:
        if init.name in names:
            array=numpy_helper.to_array(init).reshape(())
            init.CopyFrom(numpy_helper.from_array(array,name=init.name))
    output.parent.mkdir(parents=True,exist_ok=True);onnx.checker.check_model(model,full_check=True);onnx.save(model,output);return output
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();print(build(a.source,a.output))
if __name__=="__main__":main()
