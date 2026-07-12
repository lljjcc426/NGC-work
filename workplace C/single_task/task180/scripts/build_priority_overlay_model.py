from __future__ import annotations

import csv, sys
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

TASK='task180'; HERE=Path(__file__).resolve(); TASK_DIR=HERE.parents[1]; REPO=HERE.parents[4]
COMMON=REPO/'workplace C'/'neurogolf-2026-work'/'scripts'
BASE=Path(r'E:/kagglegolf/submissions/candidates/GOLF_20260711_097_v96_plus_task132_task046/onnx/task180.onnx')
OUT=TASK_DIR/'onnx'/'task180_priority_overlay.onnx'
def init(n,v): return numpy_helper.from_array(v,n)

def build_onnx(path:Path=OUT)->Path:
    path.parent.mkdir(parents=True,exist_ok=True)
    nodes=[
      helper.make_node('Slice',['input','s4','e4','axes'],['m4']),
      helper.make_node('Slice',['input','s5','e5','axes'],['m5']),
      helper.make_node('Slice',['input','s6','e6','axes'],['m6']),
      helper.make_node('Slice',['input','s9','e9','axes'],['m9']),
      helper.make_node('Equal',['m5','one_f'],['b5']),helper.make_node('Equal',['m6','one_f'],['b6']),
      helper.make_node('Equal',['m9','one_f'],['b9']),helper.make_node('Equal',['m4','one_f'],['b4']),
      helper.make_node('Not',['b5'],['n5']),helper.make_node('Not',['b6'],['n6']),helper.make_node('Not',['b9'],['n9']),helper.make_node('Not',['b4'],['n4']),
      helper.make_node('And',['n5','b6'],['o6']),
      helper.make_node('And',['n5','n6'],['free56']),helper.make_node('And',['free56','b9'],['o9']),
      helper.make_node('And',['free56','n9'],['free569']),helper.make_node('And',['free569','b4'],['o4']),
      helper.make_node('And',['free569','n4'],['o0']),
      helper.make_node('And',['b5','n5'],['z']),
      helper.make_node('Concat',['o0','z','z','z','o4','b5','o6','z','z','o9'],['small_b'],axis=1),
      helper.make_node('Cast',['small_b'],['small'],to=TensorProto.FLOAT),
      helper.make_node('Pad',['small','pads','', 'spatial_axes'],['output']),
    ]
    graph=helper.make_graph(nodes,'task180_quadrant_priority_overlay',
      [helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],
      [helper.make_tensor_value_info('output',TensorProto.FLOAT,[1,10,30,30])],
      [init('axes',np.array([1,2,3],np.int64)),
       init('s4',np.array([4,0,0],np.int64)),init('e4',np.array([5,4,4],np.int64)),
       init('s5',np.array([5,0,4],np.int64)),init('e5',np.array([6,4,8],np.int64)),
       init('s6',np.array([6,4,0],np.int64)),init('e6',np.array([7,8,4],np.int64)),
       init('s9',np.array([9,4,4],np.int64)),init('e9',np.array([10,8,8],np.int64)),
       init('one_f',np.array(1,np.float32)),init('pads',np.array([0,0,26,26],np.int64)),init('spatial_axes',np.array([2,3],np.int64))])
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',18)]); onnx.checker.check_model(model); onnx.save(model,path); return path

def main():
    sys.path.insert(0,str(COMMON)); from c_score_common import score_onnx
    c=build_onnx(); old=score_onnx(TASK,BASE,True); new=score_onnx(TASK,c,True)
    rows=[{'model':'baseline','passed':old.examples_passed,'checked':old.examples_checked,'memory':old.memory,'params':old.params,'cost':old.cost,'ok':old.ok,'artifact':str(BASE)},
          {'model':'priority_overlay','passed':new.examples_passed,'checked':new.examples_checked,'memory':new.memory,'params':new.params,'cost':new.cost,'ok':new.ok,'artifact':str(c)}]
    p=TASK_DIR/'reports'/'cost_diff.csv'; p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    print(rows)
if __name__=='__main__': main()
