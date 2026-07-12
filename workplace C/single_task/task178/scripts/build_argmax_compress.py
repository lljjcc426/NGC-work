from __future__ import annotations

import csv, sys
from pathlib import Path
import numpy as np, onnx
from onnx import TensorProto, helper, numpy_helper

TASK="task178";HERE=Path(__file__).resolve();TASK_DIR=HERE.parents[1];REPO=HERE.parents[4]
sys.path.insert(0,str(REPO/"workplace C"/"neurogolf-2026-work"/"scripts"))
from c_score_common import CURRENT_BEST_ONNX_DIR, score_onnx  # noqa:E402
def ini(n,v):return numpy_helper.from_array(np.asarray(v),name=n)

def build(path:Path)->Path:
 path.parent.mkdir(parents=True,exist_ok=True)
 ns=[
  helper.make_node("ArgMax",["input"],["labels_3d"],axis=1,keepdims=0),
  helper.make_node("Slice",["labels_3d","row_start","row_end","axes_3d"],["row_3d"]),helper.make_node("Reshape",["row_3d","line_shape"],["row"]),
  helper.make_node("Slice",["labels_3d","col_start","col_end","axes_3d"],["col_3d"]),helper.make_node("Reshape",["col_3d","line_shape"],["col"]),
  helper.make_node("Slice",["col","one","four"],["col_curr3"]),helper.make_node("Slice",["col","zero","three"],["col_prev3"]),
  helper.make_node("Equal",["col_curr3","col_prev3"],["col_eq"]),helper.make_node("Not",["col_eq"],["col_neq"]),helper.make_node("Cast",["col_curr3"],["col_valid"] ,to=TensorProto.BOOL),
  helper.make_node("And",["col_neq","col_valid"],["vertical_changes"]),helper.make_node("Cast",["vertical_changes"],["vertical_u8"],to=TensorProto.UINT8),
  helper.make_node("ReduceMax",["vertical_u8"],["vertical_u8_scalar"],keepdims=0),helper.make_node("Cast",["vertical_u8_scalar"],["vertical"],to=TensorProto.BOOL),
  helper.make_node("Where",["vertical","col","row"],["selected"]),
  helper.make_node("Slice",["selected","one","thirteen"],["curr"]),helper.make_node("Slice",["selected","zero","twelve"],["prev"]),
  helper.make_node("Equal",["curr","prev"],["same"]),helper.make_node("Not",["same"],["changed"]),helper.make_node("Cast",["curr"],["curr_valid"],to=TensorProto.BOOL),
  helper.make_node("And",["changed","curr_valid"],["tail_starts"]),helper.make_node("Cast",["selected"],["selected_valid"],to=TensorProto.BOOL),helper.make_node("Slice",["selected_valid","zero","one"],["first_start"]),
  helper.make_node("Concat",["first_start","tail_starts"],["starts"],axis=0),helper.make_node("Where",["starts","descending","zero_f"],["scores"]),
  helper.make_node("TopK",["scores","k5"],["top_values","top_idx"],axis=0),helper.make_node("Gather",["selected","top_idx"],["top_labels"],axis=0),
  helper.make_node("Gather",["starts","top_idx"],["top_valid"],axis=0),helper.make_node("Where",["top_valid","top_labels","invalid_label"],["labels5"]),
  helper.make_node("Reshape",["labels5","compressed_shape"],["compressed_3d"]),helper.make_node("Equal",["channel_ids","compressed_3d"],["onehot"]),
  helper.make_node("Reshape",["onehot","row_shape"],["answer_row"]),helper.make_node("Pad",["answer_row","row_pads"],["answer_row_wide"]),helper.make_node("Slice",["answer_row_wide","crop0","crop30","crop_axes"],["row_out"]),
  helper.make_node("Reshape",["onehot","col_shape"],["answer_col"]),helper.make_node("Pad",["answer_col","col_pads"],["answer_col_tall"]),helper.make_node("Slice",["answer_col_tall","crop0","crop30","crop_axes"],["col_out"]),
  helper.make_node("Cast",["row_out"],["row_u8"],to=TensorProto.UINT8),helper.make_node("Cast",["col_out"],["col_u8"],to=TensorProto.UINT8),
  helper.make_node("Where",["vertical","col_u8","row_u8"],["selected_u8"]),helper.make_node("Cast",["selected_u8"],["output"],to=TensorProto.BOOL),
 ]
 ins=[ini("row_start",np.array([0,0,0],np.int64)),ini("row_end",np.array([1,1,13],np.int64)),ini("col_start",np.array([0,0,0],np.int64)),ini("col_end",np.array([1,13,1],np.int64)),ini("axes_3d",np.array([0,1,2],np.int64)),
      ini("line_shape",np.array([13],np.int64)),ini("zero",np.array([0],np.int64)),ini("one",np.array([1],np.int64)),ini("three",np.array([3],np.int64)),ini("four",np.array([4],np.int64)),ini("twelve",np.array([12],np.int64)),ini("thirteen",np.array([13],np.int64)),
      ini("descending",np.arange(13,0,-1,dtype=np.float16)),ini("zero_f",np.array(0,np.float16)),ini("k5",np.array([5],np.int64)),ini("invalid_label",np.array(10,np.int64)),
      ini("compressed_shape",np.array([1,1,5],np.int64)),ini("channel_ids",np.arange(10,dtype=np.int64).reshape(1,10,1)),ini("row_shape",np.array([1,10,1,5],np.int64)),ini("col_shape",np.array([1,10,5,1],np.int64)),
      ini("row_pads",np.array([0,0,0,0,0,0,29,29],np.int64)),ini("col_pads",np.array([0,0,0,0,0,0,29,29],np.int64)),ini("crop0",np.array([0,0],np.int64)),ini("crop30",np.array([30,30],np.int64)),ini("crop_axes",np.array([2,3],np.int64))]
 g=helper.make_graph(ns,"task178_argmax_compress",[helper.make_tensor_value_info("input",TensorProto.FLOAT,[1,10,30,30])],[helper.make_tensor_value_info("output",TensorProto.BOOL,[1,10,30,30])],ins)
 m=helper.make_model(g,opset_imports=[helper.make_opsetid("",18)],ir_version=8);onnx.checker.check_model(m);onnx.save(m,path);return path

def main():
 p=build(TASK_DIR/"onnx"/f"{TASK}_candidate.onnx");old=score_onnx(TASK,CURRENT_BEST_ONNX_DIR/f"{TASK}.onnx");new=score_onnx(TASK,p)
 (TASK_DIR/"reports").mkdir(parents=True,exist_ok=True)
 with (TASK_DIR/"reports"/"cost_diff.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=["task","variant","passed","checked","cost","points","valid","artifact"]);w.writeheader()
  for v,r in [("baseline",old),("argmax_compress",new)]:w.writerow({"task":TASK,"variant":v,"passed":r.examples_passed,"checked":r.examples_checked,"cost":r.cost,"points":r.points,"valid":r.ok,"artifact":r.path})
 print(old);print(new)
if __name__=="__main__":main()
