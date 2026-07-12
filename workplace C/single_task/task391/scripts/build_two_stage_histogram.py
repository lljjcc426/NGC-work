from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np,onnx
from onnx import helper,numpy_helper
def build(source:Path,output:Path)->Path:
 m=deepcopy(onnx.load(source));old=list(m.graph.node);m.opset_import[0].version=18;m.graph.initializer.extend([numpy_helper.from_array(np.array([2,3],np.int64),name='spatial_axes'),numpy_helper.from_array(np.array([0],np.int64),name='batch_axis')])
 repl=[helper.make_node('ReduceSum',['input','spatial_axes'],['batch_counts'],keepdims=0),helper.make_node('Squeeze',['batch_counts','batch_axis'],['cnt'])]
 del m.graph.node[:];m.graph.node.extend(repl+old[1:]);output.parent.mkdir(parents=True,exist_ok=True);onnx.checker.check_model(m,full_check=True);onnx.save(m,output);return output
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(build(a.source,a.output))
if __name__=='__main__':main()
