from __future__ import annotations

import csv, sys
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

TASK='task362'; HERE=Path(__file__).resolve(); TASK_DIR=HERE.parents[1]; REPO=HERE.parents[4]
COMMON=REPO/'workplace C'/'neurogolf-2026-work'/'scripts'
BASE=Path(r'E:/kagglegolf/submissions/candidates/GOLF_20260711_097_v96_plus_task132_task046/onnx/task362.onnx')
OUT=TASK_DIR/'onnx'/'task362_dynamic_shift.onnx'
def init(n,v): return numpy_helper.from_array(v,n)

def build_onnx(path:Path=OUT)->Path:
    path.parent.mkdir(parents=True,exist_ok=True)
    bgsel=np.zeros((1,10,1,1),np.float32); bgsel[:,0]=1
    nodes=[
      helper.make_node('Slice',['input','area_s','area_e','spatial_axes'],['area']),
      helper.make_node('ReduceSum',['area','spatial_axes'],['counts_b'],keepdims=0),
      helper.make_node('Squeeze',['counts_b','batch_axis'],['counts']),
      helper.make_node('Equal',['counts','nineteen'],['cross_match']),
      helper.make_node('Cast',['cross_match'],['cross_i'],to=TensorProto.INT32),
      helper.make_node('ArgMax',['cross_i'],['cross_channel'],axis=0,keepdims=1),
      helper.make_node('Gather',['counts','marker_channel'],['marker_count']),
      helper.make_node('Gather',['area','cross_channel'],['cross'],axis=1),
      helper.make_node('Einsum',['cross','coord10'],['weighted_row'],equation='bchw,h->b'),
      helper.make_node('Einsum',['cross','coord10'],['weighted_col'],equation='bchw,w->b'),
      helper.make_node('Sub',['weighted_row','forty_five'],['row_numer']),
      helper.make_node('Sub',['weighted_col','forty_five'],['col_numer']),
      helper.make_node('Div',['row_numer','nine'],['source_row']),
      helper.make_node('Div',['col_numer','nine'],['source_col']),
      helper.make_node('Add',['source_row','marker_count'],['target_row']),
      helper.make_node('Sub',['source_col','marker_count'],['target_col']),
      helper.make_node('Equal',['pos_row10','target_row'],['row_hot']),
      helper.make_node('Equal',['pos_col10','target_col'],['col_hot']),
      helper.make_node('Or',['row_hot','col_hot'],['line2d']),
      helper.make_node('Cast',['line2d'],['line_f'],to=TensorProto.FLOAT),
      helper.make_node('Unsqueeze',['line_f','axes01'],['shifted']),
      helper.make_node('OneHot',['cross_channel','depth','hot_values'],['selector10'],axis=-1),
      helper.make_node('Reshape',['selector10','selector_shape'],['selector']),
      helper.make_node('Mul',['shifted','selector'],['colored']),
      helper.make_node('Sub',['one_f','shifted'],['background']),
      helper.make_node('Mul',['background','background_selector'],['background10']),
      helper.make_node('Add',['colored','background10'],['small']),
      helper.make_node('Pad',['small','outer_pads','', 'spatial_axes'],['output']),
    ]
    graph=helper.make_graph(nodes,'task362_marker_count_shift',
      [helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],
      [helper.make_tensor_value_info('output',TensorProto.FLOAT,[1,10,30,30])],
      [init('area_s',np.array([0,0],np.int64)),init('area_e',np.array([10,10],np.int64)),init('spatial_axes',np.array([2,3],np.int64)),
       init('batch_axis',np.array([0],np.int64)),init('nineteen',np.array(19,np.float32)),init('marker_channel',np.array(5,np.int64)),
       init('channel_axis',np.array([1],np.int64)),init('axis0',np.array([0],np.int64)),init('coord10',np.arange(10,dtype=np.float32)),
       init('forty_five',np.array(45,np.float32)),init('nine',np.array(9,np.float32)),init('pos_row10',np.arange(10,dtype=np.float32).reshape(10,1)),init('pos_col10',np.arange(10,dtype=np.float32)),init('axes01',np.array([0,1],np.int64)),
       init('depth',np.array(10,np.int64)),init('hot_values',np.array([0,1],np.float32)),
       init('selector_shape',np.array([1,10,1,1],np.int64)),init('one_f',np.array(1,np.float32)),init('background_selector',bgsel),init('outer_pads',np.array([0,0,20,20],np.int64))])
    graph.value_info.extend([
      helper.make_tensor_value_info('shifted',TensorProto.FLOAT,[1,1,10,10]),
      helper.make_tensor_value_info('colored',TensorProto.FLOAT,[1,10,10,10]),
      helper.make_tensor_value_info('background',TensorProto.FLOAT,[1,1,10,10]),
      helper.make_tensor_value_info('background10',TensorProto.FLOAT,[1,10,10,10]),
      helper.make_tensor_value_info('small',TensorProto.FLOAT,[1,10,10,10]),
    ])
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',18)]); onnx.checker.check_model(model); onnx.save(model,path); return path

def main():
    sys.path.insert(0,str(COMMON)); from c_score_common import score_onnx
    c=build_onnx(); old=score_onnx(TASK,BASE,True); new=score_onnx(TASK,c,True)
    rows=[{'model':'baseline','passed':old.examples_passed,'checked':old.examples_checked,'memory':old.memory,'params':old.params,'cost':old.cost,'ok':old.ok,'artifact':str(BASE)},
          {'model':'dynamic_shift','passed':new.examples_passed,'checked':new.examples_checked,'memory':new.memory,'params':new.params,'cost':new.cost,'ok':new.ok,'artifact':str(c)}]
    p=TASK_DIR/'reports'/'cost_diff.csv'; p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    print(rows)
if __name__=='__main__': main()
