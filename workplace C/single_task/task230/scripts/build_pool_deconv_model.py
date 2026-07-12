from __future__ import annotations

import csv, sys
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

TASK='task230'; HERE=Path(__file__).resolve(); TASK_DIR=HERE.parents[1]; REPO=HERE.parents[4]
COMMON=REPO/'workplace C'/'neurogolf-2026-work'/'scripts'
BASE=Path(r'E:/kagglegolf/submissions/candidates/GOLF_20260711_097_v96_plus_task132_task046/onnx/task230.onnx')
OUT=TASK_DIR/'onnx'/'task230_pool_deconv.onnx'
def init(n,v): return numpy_helper.from_array(v,n)

def build_onnx(path:Path=OUT)->Path:
    path.parent.mkdir(parents=True,exist_ok=True)
    w=np.zeros((1,4,4,4),np.float32)
    w[0,0,0,0]=1; w[0,1,0,3]=1; w[0,2,3,0]=1; w[0,3,3,3]=1
    nodes=[
      helper.make_node('Gather',['input','five'],['five_plane'],axis=1),
      helper.make_node('AveragePool',['five_plane'],['avg2'],kernel_shape=[2,2],strides=[1,1]),
      helper.make_node('Equal',['avg2','one_f'],['blocks_b']),
      helper.make_node('Cast',['blocks_b'],['blocks'],to=TensorProto.FLOAT),
      helper.make_node('ConvTranspose',['blocks','corner_w'],['marker32'],kernel_shape=[4,4]),
      helper.make_node('Slice',['marker32','crop_s','crop_e','spatial_axes'],['markers']),
      helper.make_node('ReduceSum',['markers','channel_axis'],['marker_any'],keepdims=1),
      helper.make_node('Gather',['input','zero_channel'],['input_background'],axis=1),
      helper.make_node('Sub',['input_background','marker_any'],['background']),
      helper.make_node('Mul',['five_plane','zero_f'],['zero']),
      helper.make_node('Split',['markers'],['m1','m2','m3','m4'],axis=1,num_outputs=4),
      helper.make_node('Concat',['background','m1','m2','m3','m4','five_plane','zero','zero','zero','zero'],['output'],axis=1),
    ]
    graph=helper.make_graph(nodes,'task230_block_detector_corner_deconvolution',
      [helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],
      [helper.make_tensor_value_info('output',TensorProto.FLOAT,[1,10,30,30])],
      [init('five',np.array([5],np.int64)),init('one_f',np.array(1,np.float32)),init('corner_w',w),
       init('crop_s',np.array([1,1],np.int64)),init('crop_e',np.array([31,31],np.int64)),init('spatial_axes',np.array([2,3],np.int64)),
       init('channel_axis',np.array([1],np.int64)),init('zero_channel',np.array([0],np.int64)),init('zero_f',np.array(0,np.float32))])
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',18)]); onnx.checker.check_model(model); onnx.save(model,path); return path

def main():
    sys.path.insert(0,str(COMMON)); from c_score_common import score_onnx
    c=build_onnx(); old=score_onnx(TASK,BASE,True); new=score_onnx(TASK,c,True)
    rows=[{'model':'baseline','passed':old.examples_passed,'checked':old.examples_checked,'memory':old.memory,'params':old.params,'cost':old.cost,'ok':old.ok,'artifact':str(BASE)},
          {'model':'pool_deconv','passed':new.examples_passed,'checked':new.examples_checked,'memory':new.memory,'params':new.params,'cost':new.cost,'ok':new.ok,'artifact':str(c)}]
    p=TASK_DIR/'reports'/'cost_diff.csv'; p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    print(rows)
if __name__=='__main__': main()
