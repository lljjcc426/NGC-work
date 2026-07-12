from __future__ import annotations

import csv, sys
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

TASK='task052'; HERE=Path(__file__).resolve(); TASK_DIR=HERE.parents[1]; REPO=HERE.parents[4]
COMMON=REPO/'workplace C'/'neurogolf-2026-work'/'scripts'
BASE=Path(r'E:/kagglegolf/submissions/candidates/GOLF_20260711_097_v96_plus_task132_task046/onnx/task052.onnx')
OUT=TASK_DIR/'onnx'/'task052_monochrome_rows.onnx'
def init(n,v): return numpy_helper.from_array(v,n)

def build_onnx(path:Path=OUT)->Path:
    path.parent.mkdir(parents=True,exist_ok=True)
    weights=np.ones((10,1,1,3),np.float32)
    nodes=[
      helper.make_node('Slice',['input','area_s','area_e','spatial_axes'],['area']),
      helper.make_node('Conv',['area','row_sum_w'],['color_counts'],kernel_shape=[1,3],group=10),
      helper.make_node('Equal',['color_counts','three_f'],['full_color']),
      helper.make_node('Cast',['full_color'],['full_i'],to=TensorProto.INT32),
      helper.make_node('ReduceMax',['full_i','channel_axis'],['mono_i'],keepdims=1),
      helper.make_node('Equal',['mono_i','one_i'],['mono']),
      helper.make_node('Not',['mono'],['background']),
      helper.make_node('And',['mono','background'],['zero']),
      helper.make_node('Concat',['background','background','background'],['bg3'],axis=3),
      helper.make_node('Concat',['mono','mono','mono'],['mono3'],axis=3),
      helper.make_node('Concat',['zero','zero','zero'],['zero3'],axis=3),
      helper.make_node('Concat',['bg3','zero3','zero3','zero3','zero3','mono3','zero3','zero3','zero3','zero3'],['small'],axis=1),
      helper.make_node('Pad',['small','pads','', 'spatial_axes'],['output']),
    ]
    graph=helper.make_graph(nodes,'task052_monochromatic_row_detector',
      [helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],
      [helper.make_tensor_value_info('output',TensorProto.BOOL,[1,10,30,30])],
      [init('area_s',np.array([0,0],np.int64)),init('area_e',np.array([3,3],np.int64)),init('spatial_axes',np.array([2,3],np.int64)),
       init('row_sum_w',weights),init('three_f',np.array(3,np.float32)),init('channel_axis',np.array([1],np.int64)),init('one_i',np.array(1,np.int32)),init('pads',np.array([0,0,27,27],np.int64))])
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',18)]); onnx.checker.check_model(model); onnx.save(model,path); return path

def main():
    sys.path.insert(0,str(COMMON)); from c_score_common import score_onnx
    c=build_onnx(); old=score_onnx(TASK,BASE,True); new=score_onnx(TASK,c,True)
    rows=[{'model':'baseline','passed':old.examples_passed,'checked':old.examples_checked,'memory':old.memory,'params':old.params,'cost':old.cost,'ok':old.ok,'artifact':str(BASE)},
          {'model':'monochrome_rows','passed':new.examples_passed,'checked':new.examples_checked,'memory':new.memory,'params':new.params,'cost':new.cost,'ok':new.ok,'artifact':str(c)}]
    p=TASK_DIR/'reports'/'cost_diff.csv'; p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    print(rows)
if __name__=='__main__': main()
