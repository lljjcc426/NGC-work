from __future__ import annotations

import csv, sys
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

TASK='task121'; HERE=Path(__file__).resolve(); TASK_DIR=HERE.parents[1]; REPO=HERE.parents[4]
COMMON=REPO/'workplace C'/'neurogolf-2026-work'/'scripts'
BASE=Path(r'E:/kagglegolf/submissions/candidates/GOLF_20260711_097_v96_plus_task132_task046/onnx/task121.onnx')
OUT=TASK_DIR/'onnx'/'task121_marker_object.onnx'
def init(n,v): return numpy_helper.from_array(v,n)

def build_onnx(path:Path=OUT)->Path:
    path.parent.mkdir(parents=True,exist_ok=True)
    keep=np.ones((1,10,1,1),np.float32); keep[:,0]=0; keep[:,8]=0
    bg=np.zeros((1,10,1,1),np.float32); bg[:,0]=1
    nodes=[
      helper.make_node('Gather',['input','marker_channel'],['marker'],axis=1),
      helper.make_node('Einsum',['marker','coord'],['marker_row'],equation='bchw,h->b'),
      helper.make_node('Einsum',['marker','coord'],['marker_col'],equation='bchw,w->b'),
      helper.make_node('Cast',['marker_row'],['marker_row_i'],to=TensorProto.INT64),
      helper.make_node('Cast',['marker_col'],['marker_col_i'],to=TensorProto.INT64),
      helper.make_node('Concat',['marker_row_i','marker_col_i'],['marker_rc'],axis=0),
      helper.make_node('Sub',['marker_rc','one2'],['starts']),
      helper.make_node('Add',['starts','three2'],['ends']),
      helper.make_node('Slice',['input','starts','ends','spatial_axes'],['patch']),
      helper.make_node('Gather',['patch','background_channel'],['patch_bg'],axis=1),
      helper.make_node('Equal',['patch_bg','zero_f'],['object_mask']),
      helper.make_node('ReduceMax',['patch','spatial_axes'],['present'],keepdims=1),
      helper.make_node('Mul',['present','keep_channels'],['base_color']),
      helper.make_node('Where',['object_mask','base_color','background_selector'],['small']),
      helper.make_node('Pad',['small','pads','', 'pad_axes'],['output']),
    ]
    graph=helper.make_graph(nodes,'task121_marker_centered_object',
      [helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],
      [helper.make_tensor_value_info('output',TensorProto.FLOAT,[1,10,30,30])],
      [init('marker_channel',np.array([8],np.int64)),init('coord',np.arange(30,dtype=np.float32)),
       init('one2',np.array([1,1],np.int64)),init('three2',np.array([3,3],np.int64)),init('spatial_axes',np.array([2,3],np.int64)),
       init('background_channel',np.array([0],np.int64)),init('zero_f',np.array(0,np.float32)),init('keep_channels',keep),init('background_selector',bg),
       init('pads',np.array([0,0,27,27],np.int64)),init('pad_axes',np.array([2,3],np.int64))])
    graph.value_info.extend([
      helper.make_tensor_value_info('patch',TensorProto.FLOAT,[1,10,3,3]),
      helper.make_tensor_value_info('patch_bg',TensorProto.FLOAT,[1,1,3,3]),
      helper.make_tensor_value_info('object_mask',TensorProto.BOOL,[1,1,3,3]),
      helper.make_tensor_value_info('present',TensorProto.FLOAT,[1,10,1,1]),
      helper.make_tensor_value_info('base_color',TensorProto.FLOAT,[1,10,1,1]),
      helper.make_tensor_value_info('small',TensorProto.FLOAT,[1,10,3,3]),
    ])
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',18)]); onnx.checker.check_model(model); onnx.save(model,path); return path

def main():
    sys.path.insert(0,str(COMMON)); from c_score_common import score_onnx
    c=build_onnx(); old=score_onnx(TASK,BASE,True); new=score_onnx(TASK,c,True)
    rows=[{'model':'baseline','passed':old.examples_passed,'checked':old.examples_checked,'memory':old.memory,'params':old.params,'cost':old.cost,'ok':old.ok,'artifact':str(BASE)},
          {'model':'marker_object','passed':new.examples_passed,'checked':new.examples_checked,'memory':new.memory,'params':new.params,'cost':new.cost,'ok':new.ok,'artifact':str(c)}]
    p=TASK_DIR/'reports'/'cost_diff.csv'; p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    print(rows)
if __name__=='__main__': main()
