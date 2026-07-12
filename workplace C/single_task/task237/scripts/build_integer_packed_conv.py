from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

def build(source: Path, output: Path) -> Path:
    model=deepcopy(onnx.load(source));old=list(model.graph.node);weights=next(numpy_helper.to_array(x) for x in model.graph.initializer if x.name=="packed_kernel")
    scaled=np.rint(weights*16).astype(np.uint8);model.graph.initializer.extend([numpy_helper.from_array(scaled,name="packed_kernel_u8"),numpy_helper.from_array(np.array(16,dtype=np.float32),name="sixteen_f")])
    repl=[helper.make_node("Cast",["input"],["input_u8"],to=TensorProto.UINT8),helper.make_node("ConvInteger",["input_u8","packed_kernel_u8"],["packed_grid_i32"],strides=[1,30]),helper.make_node("Cast",["packed_grid_i32"],["packed_grid_scaled"],to=TensorProto.FLOAT),helper.make_node("Div",["packed_grid_scaled","sixteen_f"],["packed_grid30"])]
    del model.graph.node[:];model.graph.node.extend(old[:2]+repl+old[3:]);output.parent.mkdir(parents=True,exist_ok=True);onnx.checker.check_model(model,full_check=True);onnx.save(model,output);return output
def main():
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();print(build(a.source,a.output))
if __name__=="__main__":main()
