from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto,helper,numpy_helper

def build(source: Path, output: Path) -> Path:
    model=deepcopy(onnx.load(source));old=list(model.graph.node);init={x.name:numpy_helper.to_array(x) for x in model.graph.initializer}
    model.graph.initializer.extend([numpy_helper.from_array(init['frame5_w'].astype(np.float32),name='frame5_weight_f'),numpy_helper.from_array(init['fill3_w'].astype(np.float32),name='fill3_weight_f'),numpy_helper.from_array(np.array(2.5,dtype=np.float32),name='frame_threshold')])
    repl=[helper.make_node('Cast',['ch5_u8'],['ch5_f'],to=TensorProto.FLOAT),helper.make_node('Conv',['ch5_f','frame5_weight_f'],['frame_score']),helper.make_node('GreaterOrEqual',['frame_score','frame_threshold'],['frame_bool']),helper.make_node('Cast',['frame_bool'],['frame5_bin'],to=TensorProto.UINT8),helper.make_node('Cast',['frame5_bin'],['frame5_float'],to=TensorProto.FLOAT),helper.make_node('Conv',['frame5_float','fill3_weight_f'],['fill8_f'],pads=[3,3,3,3]),helper.make_node('Cast',['fill8_f'],['fill8'],to=TensorProto.UINT8)]
    del model.graph.node[:];model.graph.node.extend(old[:2]+repl+old[4:]);output.parent.mkdir(parents=True,exist_ok=True);onnx.checker.check_model(model,full_check=True);onnx.save(model,output);return output
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(build(a.source,a.output))
if __name__=='__main__':main()
