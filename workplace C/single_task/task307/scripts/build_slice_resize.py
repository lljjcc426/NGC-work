from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np,onnx
from onnx import helper,numpy_helper
def build(source:Path,output:Path)->Path:
 m=deepcopy(onnx.load(source));m.opset_import[0].version=18
 vals={'starts':np.array([0,0],np.int64),'ends':np.array([15,15],np.int64),'axes':np.array([2,3],np.int64),'roi_empty':np.array([],np.float32),'scales':np.array([1,1,2,2],np.float32)}
 for n,v in vals.items():m.graph.initializer.append(numpy_helper.from_array(v,name=n))
 nodes=[helper.make_node('Slice',['input','starts','ends','axes'],['input15']),helper.make_node('Resize',['input15','roi_empty','scales'],['output'],mode='nearest',coordinate_transformation_mode='asymmetric',nearest_mode='floor')]
 del m.graph.node[:];m.graph.node.extend(nodes);output.parent.mkdir(parents=True,exist_ok=True);onnx.checker.check_model(m,full_check=True);onnx.save(m,output);return output
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(build(a.source,a.output))
if __name__=='__main__':main()
