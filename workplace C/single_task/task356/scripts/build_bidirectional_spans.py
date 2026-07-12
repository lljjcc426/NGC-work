from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto,helper,numpy_helper

def build(source: Path, output: Path) -> Path:
    model=deepcopy(onnx.load(source));old=list(model.graph.node)
    for name,value in {'axis2':np.array(2,dtype=np.int64),'axis3':np.array(3,dtype=np.int64),'rev10':np.arange(9,-1,-1,dtype=np.int64),'zero_i32':np.array(0,dtype=np.int32)}.items():model.graph.initializer.append(numpy_helper.from_array(value,name=name))
    def scan(axis,label):
        return [helper.make_node('CumSum',['crop_i32',f'axis{axis}'],[f'{label}_forward']),helper.make_node('Gather',['crop_i32','rev10'],[f'{label}_rev'],axis=axis),helper.make_node('CumSum',[f'{label}_rev',f'axis{axis}'],[f'{label}_back_rev']),helper.make_node('Gather',[f'{label}_back_rev','rev10'],[f'{label}_back'],axis=axis),helper.make_node('Greater',[f'{label}_forward','zero_i32'],[f'{label}_seen_a']),helper.make_node('Greater',[f'{label}_back','zero_i32'],[f'{label}_seen_b']),helper.make_node('And',[f'{label}_seen_a',f'{label}_seen_b'],[f'{label}_span_bool'])]
    repl=[helper.make_node('Cast',['crop'],['crop_i32'],to=TensorProto.INT32)]+scan(3,'row')+scan(2,'col')+[helper.make_node('Or',['row_span_bool','col_span_bool'],['mask_bool']),helper.make_node('Cast',['mask_bool'],['mask10'],to=TensorProto.UINT8)]
    del model.graph.node[:];model.graph.node.extend(old[:2]+repl+old[9:]);output.parent.mkdir(parents=True,exist_ok=True);onnx.checker.check_model(model,full_check=True);onnx.save(model,output);return output
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(build(a.source,a.output))
if __name__=='__main__':main()
