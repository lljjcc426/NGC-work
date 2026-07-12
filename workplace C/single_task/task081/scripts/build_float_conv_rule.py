from __future__ import annotations

import csv, sys
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

TASK='task081'; HERE=Path(__file__).resolve(); TASK_DIR=HERE.parents[1]; REPO=HERE.parents[4]
COMMON=REPO/'workplace C'/'neurogolf-2026-work'/'scripts'
BASE=Path(r'E:/kagglegolf/submissions/candidates/GOLF_20260711_097_v96_plus_task132_task046/onnx/task081.onnx')
OUT=TASK_DIR/'onnx'/'task081_float_conv.onnx'
def init(n,v): return numpy_helper.from_array(v,n)

def build_onnx(path:Path=OUT)->Path:
    path.parent.mkdir(parents=True,exist_ok=True)
    hidden=np.array([[[[0,0,0],[6,-6,6],[0,12,0]]],[[[0,12,0],[6,-6,6],[0,0,0]]],[[[0,0,0],[0,3,0],[0,0,0]]]],np.float32)
    hb=np.array([-15,-15,0],np.float32)
    final=np.zeros((10,3,1,1),np.float32)
    final[0,:,0,0]=[-1,-1,-1]; final[1,:,0,0]=[1,1,-1]; final[8,:,0,0]=[-1,0,1]
    nodes=[
      helper.make_node('Slice',['input','starts','ends','axes'],['cyan']),
      helper.make_node('Conv',['cyan','hidden_w','hidden_b'],['hidden_raw'],kernel_shape=[3,3],pads=[1,1,1,1]),
      helper.make_node('Clip',['hidden_raw','clip_min','clip_max'],['hidden']),
      helper.make_node('Sub',['hidden','one'],['centered']),
      helper.make_node('Conv',['centered','final_w'],['small_raw'],kernel_shape=[1,1]),
      helper.make_node('Clip',['small_raw','clip_min','clip_max'],['small_clip']),
      helper.make_node('Cast',['small_clip'],['small'],to=TensorProto.UINT8),
      helper.make_node('Pad',['small','pads','pad_value','pad_axes'],['output']),
    ]
    graph=helper.make_graph(nodes,'task081_missing_corner_float_network',
      [helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],
      [helper.make_tensor_value_info('output',TensorProto.UINT8,[1,10,30,30])],
      [init('starts',np.array([0,8,0,0],np.int64)),init('ends',np.array([1,9,7,7],np.int64)),init('axes',np.array([0,1,2,3],np.int64)),
       init('hidden_w',hidden),init('hidden_b',hb),init('final_w',final),init('clip_min',np.array(0,np.float32)),init('clip_max',np.array(255,np.float32)),init('one',np.array(1,np.float32)),
       init('pads',np.array([0,0,23,23],np.int64)),init('pad_value',np.array(0,np.uint8)),init('pad_axes',np.array([2,3],np.int64))])
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',18)]); onnx.checker.check_model(model); onnx.save(model,path); return path

def main():
    sys.path.insert(0,str(COMMON)); from c_score_common import score_onnx
    c=build_onnx(); old=score_onnx(TASK,BASE,True); new=score_onnx(TASK,c,True)
    rows=[{'model':'baseline','passed':old.examples_passed,'checked':old.examples_checked,'memory':old.memory,'params':old.params,'cost':old.cost,'ok':old.ok,'artifact':str(BASE)},
          {'model':'float_conv_rule','passed':new.examples_passed,'checked':new.examples_checked,'memory':new.memory,'params':new.params,'cost':new.cost,'ok':new.ok,'artifact':str(c)}]
    p=TASK_DIR/'reports'/'cost_diff.csv'; p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    print(rows)
if __name__=='__main__': main()
