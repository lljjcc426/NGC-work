from __future__ import annotations

import csv, sys
from pathlib import Path
import numpy as np, onnx
from onnx import TensorProto, helper, numpy_helper

TASK="task221";HERE=Path(__file__).resolve();TASK_DIR=HERE.parents[1];REPO=HERE.parents[4]
sys.path.insert(0,str(REPO/"workplace C"/"neurogolf-2026-work"/"scripts"))
from c_score_common import CURRENT_BEST_ONNX_DIR, score_onnx  # noqa:E402
def ini(n,v):return numpy_helper.from_array(np.asarray(v),name=n)

def build(path:Path)->Path:
 path.parent.mkdir(parents=True,exist_ok=True)
 br=(np.arange(30)//3).astype(np.float32).reshape(1,1,30,1);bc=(np.arange(30)//3).astype(np.float32).reshape(1,1,1,30)
 ns=[
  helper.make_node("Slice",["input","crop_start","crop_end","axes_hw"],["motif"]),
  helper.make_node("Slice",["motif","c0_start","c0_end","axis_c"],["motif_c0"]),
  helper.make_node("ReduceSum",["motif_c0","axes_all"],["voids"],keepdims=0),helper.make_node("Sub",["nine","voids"],["filled"]),
  helper.make_node("Mul",["block_row","voids"],["row_rank"]),helper.make_node("Add",["row_rank","block_col"],["rank"]),
  helper.make_node("Less",["rank","filled"],["before_filled"]),helper.make_node("Less",["block_row","voids"],["row_inside"]),helper.make_node("Less",["block_col","voids"],["col_inside"]),
  helper.make_node("And",["row_inside","col_inside"],["inside"]),helper.make_node("And",["inside","before_filled"],["active"]),
  helper.make_node("Tile",["motif","repeats"],["motif_tiled"]),helper.make_node("Where",["active","motif_tiled","channel0"],["inside_values"]),
  helper.make_node("Where",["inside","inside_values","zero"],["output"]),
 ]
 ins=[ini("crop_start",np.array([0,0],np.int64)),ini("crop_end",np.array([3,3],np.int64)),ini("axes_hw",np.array([2,3],np.int64)),
      ini("c0_start",np.array([0],np.int64)),ini("c0_end",np.array([1],np.int64)),ini("axis_c",np.array([1],np.int64)),ini("axes_all",np.array([0,1,2,3],np.int64)),
      ini("nine",np.array(9,np.float32)),ini("block_row",br),ini("block_col",bc),ini("repeats",np.array([1,1,10,10],np.int64)),
      ini("channel0",np.eye(10,dtype=np.float32)[0].reshape(1,10,1,1)),ini("zero",np.array(0,np.float32))]
 g=helper.make_graph(ns,"task221_tile_rank_fill",[helper.make_tensor_value_info("input",TensorProto.FLOAT,[1,10,30,30])],[helper.make_tensor_value_info("output",TensorProto.FLOAT,[1,10,30,30])],ins)
 m=helper.make_model(g,opset_imports=[helper.make_opsetid("",18)],ir_version=8);onnx.checker.check_model(m);onnx.save(m,path);return path

def main():
 p=build(TASK_DIR/"onnx"/f"{TASK}_candidate.onnx");old=score_onnx(TASK,CURRENT_BEST_ONNX_DIR/f"{TASK}.onnx");new=score_onnx(TASK,p)
 (TASK_DIR/"reports").mkdir(parents=True,exist_ok=True)
 with (TASK_DIR/"reports"/"cost_diff.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=["task","variant","passed","checked","cost","points","valid","artifact"]);w.writeheader()
  for v,r in [("baseline",old),("tile_rank_fill",new)]:w.writerow({"task":TASK,"variant":v,"passed":r.examples_passed,"checked":r.examples_checked,"cost":r.cost,"points":r.points,"valid":r.ok,"artifact":r.path})
 print(old);print(new)
if __name__=="__main__":main()
