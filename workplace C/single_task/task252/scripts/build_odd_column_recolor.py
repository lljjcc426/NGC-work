from __future__ import annotations

import csv, sys
from pathlib import Path
import numpy as np, onnx
from onnx import TensorProto, helper, numpy_helper

TASK="task252";HERE=Path(__file__).resolve();TASK_DIR=HERE.parents[1];REPO=HERE.parents[4]
sys.path.insert(0,str(REPO/"workplace C"/"neurogolf-2026-work"/"scripts"))
from c_score_common import CURRENT_BEST_ONNX_DIR, score_onnx  # noqa:E402
def ini(n,v):return numpy_helper.from_array(np.asarray(v),name=n)

def build(path:Path)->Path:
 path.parent.mkdir(parents=True,exist_ok=True)
 even=(np.arange(30)%2==0).astype(np.float32).reshape(1,1,1,30); odd=1.0-even
 ns=[helper.make_node("Slice",["input","c0_start","c0_end","axis_c"],["background_1c"],name="extract_background"),
     helper.make_node("Pad",["background_1c","bg_pads"],["background"],name="restore_background_channels"),
     helper.make_node("Slice",["input","fg_start","fg_end","axis_c"],["foreground_9c"],name="extract_foreground"),
     helper.make_node("Mul",["foreground_9c","even_mask"],["foreground_even_9c"],name="keep_even_columns"),
     helper.make_node("Pad",["foreground_even_9c","fg_pads"],["foreground_even"],name="restore_foreground_channels"),
     helper.make_node("ReduceSum",["foreground_9c","axis_c"],["occupied"],name="collapse_foreground_colors",keepdims=1),
     helper.make_node("Mul",["occupied","odd_mask"],["odd_occupied"],name="select_odd_columns"),
     helper.make_node("Pad",["odd_occupied","color4_pads"],["recolored_four"],name="place_in_color4_channel"),
     helper.make_node("Add",["background","foreground_even"],["base_output"]),helper.make_node("Add",["base_output","recolored_four"],["output"],name="recolor_output")]
 ins=[ini("c0_start",np.array([0],np.int64)),ini("c0_end",np.array([1],np.int64)),ini("fg_start",np.array([1],np.int64)),ini("fg_end",np.array([10],np.int64)),ini("axis_c",np.array([1],np.int64)),
      ini("even_mask",even),ini("odd_mask",odd),ini("bg_pads",np.array([0,0,0,0,0,9,0,0],np.int64)),ini("fg_pads",np.array([0,1,0,0,0,0,0,0],np.int64)),ini("color4_pads",np.array([0,4,0,0,0,5,0,0],np.int64))]
 g=helper.make_graph(ns,"task252_odd_column_recolor",[helper.make_tensor_value_info("input",TensorProto.FLOAT,[1,10,30,30])],[helper.make_tensor_value_info("output",TensorProto.FLOAT,[1,10,30,30])],ins)
 m=helper.make_model(g,opset_imports=[helper.make_opsetid("",18)],ir_version=8);onnx.checker.check_model(m);onnx.save(m,path);return path

def main():
 p=build(TASK_DIR/"onnx"/f"{TASK}_candidate.onnx");old=score_onnx(TASK,CURRENT_BEST_ONNX_DIR/f"{TASK}.onnx");new=score_onnx(TASK,p)
 (TASK_DIR/"reports").mkdir(parents=True,exist_ok=True)
 with (TASK_DIR/"reports"/"cost_diff.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=["task","variant","passed","checked","memory","params","cost","points","valid","artifact"]);w.writeheader()
  for v,r in [("baseline",old),("odd_column_recolor",new)]:w.writerow({"task":TASK,"variant":v,"passed":r.examples_passed,"checked":r.examples_checked,"memory":r.memory,"params":r.params,"cost":r.cost,"points":r.points,"valid":r.ok,"artifact":r.path})
 print(old);print(new)
if __name__=="__main__":main()
