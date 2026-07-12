from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np,onnx
from onnx import TensorProto,helper,numpy_helper
def build(source:Path,output:Path)->Path:
 m=deepcopy(onnx.load(source));old=list(m.graph.node);weights={x.name:numpy_helper.to_array(x) for x in m.graph.initializer};new=[];added=set()
 for i,node in enumerate(old):
  if node.op_type!='MatMulInteger':new.append(node);continue
  a,w=node.input[:2];out=node.output[0];wf=w+'_float'
  if wf not in added:m.graph.initializer.append(numpy_helper.from_array(weights[w].astype(np.float32),name=wf));added.add(wf)
  new.extend([helper.make_node('Cast',[a],[out+'_a_float'],to=TensorProto.FLOAT),helper.make_node('MatMul',[out+'_a_float',wf],[out+'_float']),helper.make_node('Cast',[out+'_float'],[out],to=TensorProto.INT32)])
 del m.graph.node[:];m.graph.node.extend(new);output.parent.mkdir(parents=True,exist_ok=True);onnx.checker.check_model(m,full_check=True);onnx.save(m,output);return output
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(build(a.source,a.output))
if __name__=='__main__':main()
