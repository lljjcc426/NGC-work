from __future__ import annotations

import csv, sys
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

TASK='task065'; HERE=Path(__file__).resolve(); TASK_DIR=HERE.parents[1]; REPO=HERE.parents[4]
COMMON=REPO/'workplace C'/'neurogolf-2026-work'/'scripts'
BASE=Path(r'E:/kagglegolf/submissions/candidates/GOLF_20260711_097_v96_plus_task132_task046/onnx/task065.onnx')
OUT=TASK_DIR/'onnx'/'task065_fold_scatter.onnx'
def init(n,v): return numpy_helper.from_array(v,n)

def build_onnx(path:Path=OUT)->Path:
    path.parent.mkdir(parents=True,exist_ok=True)
    nodes=[
      helper.make_node('ReduceSum',['input','spatial_axes'],['counts'],keepdims=0),
      helper.make_node('ReduceSum',['counts','count_axes'],['total'],keepdims=0),
      helper.make_node('Sqrt',['total'],['side']),
      helper.make_node('Sub',['side','one_f'],['side_minus_one']),
      helper.make_node('Mul',['side_minus_one','half'],['m_float']),
      helper.make_node('Cast',['m_float'],['m'],to=TensorProto.INT64),
      helper.make_node('Slice',['input','zero2','seven2','spatial_axes'],['base7']),
      helper.make_node('Less',['pos_row7','m_float'],['valid_rows']),
      helper.make_node('Less',['pos_col7','m_float'],['valid_cols']),
      helper.make_node('And',['valid_rows','valid_cols'],['valid2d']),
      helper.make_node('Cast',['valid2d'],['valid_f'],to=TensorProto.FLOAT),
      helper.make_node('Unsqueeze',['valid_f','axes01'],['valid']),
      helper.make_node('Mul',['base7','valid'],['base']),
      helper.make_node('Equal',['counts','one_f'],['rare_match_b']),
      helper.make_node('Cast',['rare_match_b'],['rare_match'],to=TensorProto.INT32),
      helper.make_node('ArgMax',['rare_match'],['rare_channel'],axis=1,keepdims=0),
      helper.make_node('Gather',['input','rare_channel'],['rare_map'],axis=1),
      helper.make_node('Einsum',['rare_map','coord'],['rare_row_f'],equation='bchw,h->b'),
      helper.make_node('Einsum',['rare_map','coord'],['rare_col_f'],equation='bchw,w->b'),
      helper.make_node('Cast',['rare_row_f'],['rare_row'],to=TensorProto.INT64),
      helper.make_node('Cast',['rare_col_f'],['rare_col'],to=TensorProto.INT64),
      helper.make_node('Concat',['rare_row','rare_col'],['rare_rc'],axis=0),
      helper.make_node('Add',['m','one_i'],['period']),
      helper.make_node('Mod',['rare_rc','period'],['target_rc']),
      helper.make_node('Transpose',['base'],['base_nhwc'],perm=[0,2,3,1]),
      helper.make_node('Concat',['zero1','target_rc'],['target3'],axis=0),
      helper.make_node('Unsqueeze',['target3','axis0'],['scatter_index']),
      helper.make_node('OneHot',['rare_channel','depth','hot_values'],['rare_selector'],axis=-1),
      helper.make_node('ScatterND',['base_nhwc','scatter_index','rare_selector'],['folded_nhwc']),
      helper.make_node('Transpose',['folded_nhwc'],['folded'],perm=[0,3,1,2]),
      helper.make_node('Pad',['folded','outer_pads','', 'spatial_axes'],['output']),
    ]
    graph=helper.make_graph(nodes,'task065_fold_quadrants_scatter_rare_dot',
      [helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],
      [helper.make_tensor_value_info('output',TensorProto.FLOAT,[1,10,30,30])],
      [init('spatial_axes',np.array([2,3],np.int64)),init('count_axes',np.array([0,1],np.int64)),init('one_f',np.array(1,np.float32)),init('half',np.array(.5,np.float32)),
       init('axis0',np.array([0],np.int64)),init('axis1',np.array([1],np.int64)),init('zero2',np.array([0,0],np.int64)),init('zero1',np.array([0],np.int64)),init('seven2',np.array([7,7],np.int64)),
       init('pos_row7',np.arange(7,dtype=np.float32).reshape(7,1)),init('pos_col7',np.arange(7,dtype=np.float32)),init('axes01',np.array([0,1],np.int64)),
       init('coord',np.arange(30,dtype=np.float32)),init('one_i',np.array(1,np.int64)),init('depth',np.array(10,np.int64)),init('hot_values',np.array([0,1],np.float32)),init('outer_pads',np.array([0,0,23,23],np.int64))])
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',18)]); onnx.checker.check_model(model); onnx.save(model,path); return path

def main():
    sys.path.insert(0,str(COMMON)); from c_score_common import score_onnx
    c=build_onnx(); old=score_onnx(TASK,BASE,True); new=score_onnx(TASK,c,True)
    rows=[{'model':'baseline','passed':old.examples_passed,'checked':old.examples_checked,'memory':old.memory,'params':old.params,'cost':old.cost,'ok':old.ok,'artifact':str(BASE)},
          {'model':'fold_scatter','passed':new.examples_passed,'checked':new.examples_checked,'memory':new.memory,'params':new.params,'cost':new.cost,'ok':new.ok,'artifact':str(c)}]
    p=TASK_DIR/'reports'/'cost_diff.csv'; p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    print(rows)
if __name__=='__main__': main()
