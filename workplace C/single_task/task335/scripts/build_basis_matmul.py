from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np,onnx
from onnx import helper,numpy_helper
def build(source:Path,output:Path)->Path:
 m=deepcopy(onnx.load(source));old=list(m.graph.node);T=next(numpy_helper.to_array(x) for x in m.graph.initializer if x.name=='T').astype(np.float16).reshape(10,16);m.graph.initializer.extend([numpy_helper.from_array(T,name='T_flat'),numpy_helper.from_array(np.array([16,900],np.int64),name='basis_shape'),numpy_helper.from_array(np.array([1,10,30,30],np.int64),name='output_shape')])
 repl=[helper.make_node('Einsum',['A','B'],['basis4'],equation='ir,jc->ijrc'),helper.make_node('Reshape',['basis4','basis_shape'],['basis_flat']),helper.make_node('MatMul',['T_flat','basis_flat'],['output_flat']),helper.make_node('Reshape',['output_flat','output_shape'],['output'])]
 del m.graph.node[:];m.graph.node.extend(old[:32]+repl);output.parent.mkdir(parents=True,exist_ok=True);onnx.checker.check_model(m,full_check=True);onnx.save(m,output);return output
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(build(a.source,a.output))
if __name__=='__main__':main()
