from __future__ import annotations

import csv, sys
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

TASK='task108'; HERE=Path(__file__).resolve(); TASK_DIR=HERE.parents[1]; REPO=HERE.parents[4]
COMMON=REPO/'workplace C'/'neurogolf-2026-work'/'scripts'
BASE=Path(r'E:/kagglegolf/submissions/candidates/GOLF_20260711_097_v96_plus_task132_task046/onnx/task108.onnx')
OUT=TASK_DIR/'onnx'/'task108_slice_resize.onnx'
def init(n,v): return numpy_helper.from_array(v,n)

def build_onnx(path:Path=OUT)->Path:
    path.parent.mkdir(parents=True,exist_ok=True)
    nodes=[
      helper.make_node('Slice',['input','starts','ends','axes','steps'],['sampled']),
      helper.make_node('Resize',['sampled','','scales'],['large'],mode='nearest',coordinate_transformation_mode='asymmetric',nearest_mode='floor'),
      helper.make_node('Pad',['large','pads','', 'spatial_axes'],['output']),
    ]
    graph=helper.make_graph(nodes,'task108_odd_lattice_nearest_resize',
      [helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],
      [helper.make_tensor_value_info('output',TensorProto.FLOAT,[1,10,30,30])],
      [init('starts',np.array([1,1],np.int64)),init('ends',np.array([10,10],np.int64)),init('axes',np.array([2,3],np.int64)),init('steps',np.array([2,2],np.int64)),
       init('scales',np.array([1,1,4,4],np.float32)),init('pads',np.array([0,0,10,10],np.int64)),init('spatial_axes',np.array([2,3],np.int64))])
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',18)]); onnx.checker.check_model(model); onnx.save(model,path); return path

def main():
    sys.path.insert(0,str(COMMON)); from c_score_common import score_onnx
    c=build_onnx(); old=score_onnx(TASK,BASE,True); new=score_onnx(TASK,c,True)
    rows=[{'model':'baseline','passed':old.examples_passed,'checked':old.examples_checked,'memory':old.memory,'params':old.params,'cost':old.cost,'ok':old.ok,'artifact':str(BASE)},
          {'model':'slice_resize','passed':new.examples_passed,'checked':new.examples_checked,'memory':new.memory,'params':new.params,'cost':new.cost,'ok':new.ok,'artifact':str(c)}]
    p=TASK_DIR/'reports'/'cost_diff.csv'; p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    print(rows)
if __name__=='__main__': main()
