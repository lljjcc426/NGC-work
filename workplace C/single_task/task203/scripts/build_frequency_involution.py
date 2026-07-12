from __future__ import annotations

import csv, sys
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

TASK='task203'; HERE=Path(__file__).resolve(); TASK_DIR=HERE.parents[1]; REPO=HERE.parents[4]
COMMON=REPO/'workplace C'/'neurogolf-2026-work'/'scripts'
BASE=Path(r'E:/kagglegolf/submissions/candidates/GOLF_20260711_097_v96_plus_task132_task046/onnx/task203.onnx')
OUT=TASK_DIR/'onnx'/'task203_frequency_involution.onnx'
def init(n,v): return numpy_helper.from_array(v,n)

def build_onnx(path:Path=OUT)->Path:
    path.parent.mkdir(parents=True,exist_ok=True)
    nodes=[
        helper.make_node('ReduceSum',['input','spatial_axes'],['counts_b'],keepdims=0),
        helper.make_node('Squeeze',['counts_b','batch_axis'],['counts']),
        helper.make_node('ReduceMax',['counts','channel_axis'],['largest'],keepdims=0),
        helper.make_node('Add',['largest','four'],['pair_sum']),
        helper.make_node('Sub',['pair_sum','counts'],['target']),
        helper.make_node('Unsqueeze',['target','axis1'],['target_col']),
        helper.make_node('Unsqueeze',['counts','axis0'],['counts_row']),
        helper.make_node('Equal',['target_col','counts_row'],['matches']),
        helper.make_node('Cast',['matches'],['match_i'],to=TensorProto.INT32),
        helper.make_node('ArgMax',['match_i'],['remap'],axis=1,keepdims=0),
        helper.make_node('Gather',['input','remap'],['output'],axis=1),
    ]
    graph=helper.make_graph(nodes,'task203_frequency_involution',
      [helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],
      [helper.make_tensor_value_info('output',TensorProto.FLOAT,[1,10,30,30])],
      [init('spatial_axes',np.array([2,3],np.int64)),init('batch_axis',np.array([0],np.int64)),init('channel_axis',np.array([0],np.int64)),
       init('four',np.array(4,np.float32)),init('axis1',np.array([1],np.int64)),init('axis0',np.array([0],np.int64))])
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',18)]); onnx.checker.check_model(model); onnx.save(model,path); return path

def main():
    sys.path.insert(0,str(COMMON)); from c_score_common import score_onnx
    c=build_onnx(); old=score_onnx(TASK,BASE,True); new=score_onnx(TASK,c,True)
    rows=[{'model':'baseline','passed':old.examples_passed,'checked':old.examples_checked,'memory':old.memory,'params':old.params,'cost':old.cost,'ok':old.ok,'artifact':str(BASE)},
          {'model':'frequency_involution','passed':new.examples_passed,'checked':new.examples_checked,'memory':new.memory,'params':new.params,'cost':new.cost,'ok':new.ok,'artifact':str(c)}]
    p=TASK_DIR/'reports'/'cost_diff.csv'; p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    print(rows)
if __name__=='__main__': main()
