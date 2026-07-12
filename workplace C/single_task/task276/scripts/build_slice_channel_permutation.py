from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np,onnx
from onnx import helper,numpy_helper
def build(source:Path,output:Path)->Path:
 m=deepcopy(onnx.load(source));order=[0,1,6,3,4,5,2,7,8,9];m.graph.initializer.append(numpy_helper.from_array(np.array([1],np.int64),name='channel_axis'));nodes=[]
 for i,ch in enumerate(order):
  for suffix,val in [('s',ch),('e',ch+1)]:m.graph.initializer.append(numpy_helper.from_array(np.array([val],np.int64),name=f'{suffix}{i}'))
  nodes.append(helper.make_node('Slice',['input',f's{i}',f'e{i}','channel_axis'],[f'channel_{i}']))
 nodes.append(helper.make_node('Concat',[f'channel_{i}' for i in range(10)],['output'],axis=1));del m.graph.node[:];m.graph.node.extend(nodes);output.parent.mkdir(parents=True,exist_ok=True);onnx.checker.check_model(m,full_check=True);onnx.save(m,output);return output
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(build(a.source,a.output))
if __name__=='__main__':main()
