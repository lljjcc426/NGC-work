from __future__ import annotations

import csv, sys
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

TASK='task315'; HERE=Path(__file__).resolve(); TASK_DIR=HERE.parents[1]; REPO=HERE.parents[4]
COMMON=REPO/'workplace C'/'neurogolf-2026-work'/'scripts'
BASE=Path(r'E:/kagglegolf/submissions/candidates/GOLF_20260711_097_v96_plus_task132_task046/onnx/task315.onnx')
OUT=TASK_DIR/'onnx'/'task315_kronecker.onnx'
def init(n,v): return numpy_helper.from_array(v,n)

def build_onnx(path:Path=OUT)->Path:
    path.parent.mkdir(parents=True,exist_ok=True)
    bg=np.zeros((1,3,1,1,1,1),np.float32); bg[:,0]=1
    nodes=[
      helper.make_node('Slice',['input','area_s','area_e','area_axes'],['pattern']),
      helper.make_node('Gather',['pattern','two'],['mask2d'],axis=1),
      helper.make_node('Reshape',['pattern','pattern_shape'],['pattern6']),
      helper.make_node('Reshape',['mask2d','mask_shape'],['mask6']),
      helper.make_node('Equal',['mask6','one_f'],['active']),
      helper.make_node('Where',['active','pattern6','background6'],['blocks']),
      helper.make_node('Transpose',['blocks'],['ordered'],perm=[0,1,2,4,3,5]),
      helper.make_node('Reshape',['ordered','small_shape'],['small3']),
      helper.make_node('Pad',['small3','pads','', 'pad_axes'],['output']),
    ]
    graph=helper.make_graph(nodes,'task315_mask_pattern_kronecker',
      [helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],
      [helper.make_tensor_value_info('output',TensorProto.FLOAT,[1,10,30,30])],
      [init('area_s',np.array([0,0,0],np.int64)),init('area_e',np.array([3,3,3],np.int64)),init('area_axes',np.array([1,2,3],np.int64)),init('two',np.array([2],np.int64)),
       init('pattern_shape',np.array([1,3,1,1,3,3],np.int64)),init('mask_shape',np.array([1,1,3,3,1,1],np.int64)),init('one_f',np.array(1,np.float32)),init('background6',bg),
       init('small_shape',np.array([1,3,9,9],np.int64)),init('pads',np.array([0,0,0,7,21,21],np.int64)),init('pad_axes',np.array([1,2,3],np.int64))])
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',18)]); onnx.checker.check_model(model); onnx.save(model,path); return path

def main():
    sys.path.insert(0,str(COMMON)); from c_score_common import score_onnx
    c=build_onnx(); old=score_onnx(TASK,BASE,True); new=score_onnx(TASK,c,True)
    rows=[{'model':'baseline','passed':old.examples_passed,'checked':old.examples_checked,'memory':old.memory,'params':old.params,'cost':old.cost,'ok':old.ok,'artifact':str(BASE)},
          {'model':'kronecker','passed':new.examples_passed,'checked':new.examples_checked,'memory':new.memory,'params':new.params,'cost':new.cost,'ok':new.ok,'artifact':str(c)}]
    p=TASK_DIR/'reports'/'cost_diff.csv'; p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    print(rows)
if __name__=='__main__': main()
