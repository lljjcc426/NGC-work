from __future__ import annotations

import csv, sys
from pathlib import Path
import numpy as np, onnx
from onnx import TensorProto, helper, numpy_helper

TASK="task373";HERE=Path(__file__).resolve();TASK_DIR=HERE.parents[1];REPO=HERE.parents[4]
sys.path.insert(0,str(REPO/"workplace C"/"neurogolf-2026-work"/"scripts"))
from c_score_common import CURRENT_BEST_ONNX_DIR, score_onnx  # noqa:E402
def ini(n,v):return numpy_helper.from_array(np.asarray(v),name=n)

def build(path:Path)->Path:
 path.parent.mkdir(parents=True,exist_ok=True)
 ns=[helper.make_node("Slice",["input","a_start","a_end","axes_hw"],["a"],name="first_color"),
     helper.make_node("Slice",["input","b_start","b_end","axes_hw"],["b"],name="second_color"),
     helper.make_node("Concat",["a","b","a","b","a","b"],["row_ab"],name="alternating_first_row",axis=3),
     helper.make_node("Concat",["b","a","b","a","b","a"],["row_ba"],name="alternating_second_row",axis=3),
     helper.make_node("Concat",["row_ab","row_ba"],["checker"],name="checkerboard",axis=2),
     helper.make_node("Pad",["checker","pads"],["output"],name="pad_benchmark")]
 ins=[ini("a_start",np.array([0,0],np.int64)),ini("a_end",np.array([1,1],np.int64)),ini("b_start",np.array([1,0],np.int64)),ini("b_end",np.array([2,1],np.int64)),ini("axes_hw",np.array([2,3],np.int64)),
      ini("pads",np.array([0,0,0,0,0,0,28,24],np.int64))]
 g=helper.make_graph(ns,"task373_checkerboard_concat",[helper.make_tensor_value_info("input",TensorProto.FLOAT,[1,10,30,30])],[helper.make_tensor_value_info("output",TensorProto.FLOAT,[1,10,30,30])],ins)
 m=helper.make_model(g,opset_imports=[helper.make_opsetid("",18)],ir_version=8);onnx.checker.check_model(m);onnx.save(m,path);return path

def main():
 p=build(TASK_DIR/"onnx"/f"{TASK}_candidate.onnx");old=score_onnx(TASK,CURRENT_BEST_ONNX_DIR/f"{TASK}.onnx");new=score_onnx(TASK,p)
 (TASK_DIR/"reports").mkdir(parents=True,exist_ok=True)
 with (TASK_DIR/"reports"/"cost_diff.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=["task","variant","passed","checked","memory","params","cost","points","valid","artifact"]);w.writeheader()
  for v,r in [("baseline",old),("checkerboard_concat",new)]:w.writerow({"task":TASK,"variant":v,"passed":r.examples_passed,"checked":r.examples_checked,"memory":r.memory,"params":r.params,"cost":r.cost,"points":r.points,"valid":r.ok,"artifact":r.path})
 print(old);print(new)
if __name__=="__main__":main()
