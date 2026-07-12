from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np, onnx
from onnx import helper,numpy_helper
def build(source:Path,output:Path)->Path:
 m=deepcopy(onnx.load(source));w=numpy_helper.to_array(m.graph.initializer[0]);m.graph.initializer.extend([numpy_helper.from_array(w[:,:5],name='W_left'),numpy_helper.from_array(w[:,5:],name='W_right')])
 for name,val in {'s0':np.array([0],np.int64),'e5':np.array([5],np.int64),'s5':np.array([5],np.int64),'e10':np.array([10],np.int64),'ax1':np.array([1],np.int64)}.items():m.graph.initializer.append(numpy_helper.from_array(val,name=name))
 nodes=[helper.make_node('Slice',['input','s0','e5','ax1'],['input_left']),helper.make_node('Slice',['input','s5','e10','ax1'],['input_right']),helper.make_node('Conv',['input_left','W_left'],['left_out'],pads=[1,1,1,1]),helper.make_node('Conv',['input_right','W_right'],['right_out'],pads=[1,1,1,1]),helper.make_node('Add',['left_out','right_out'],['output'])]
 del m.graph.node[:];m.graph.node.extend(nodes);output.parent.mkdir(parents=True,exist_ok=True);onnx.checker.check_model(m,full_check=True);onnx.save(m,output);return output
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(build(a.source,a.output))
if __name__=='__main__':main()
