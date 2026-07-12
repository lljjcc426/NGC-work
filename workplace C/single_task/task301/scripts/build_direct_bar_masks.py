from __future__ import annotations

import csv, sys
from pathlib import Path
import numpy as np, onnx
from onnx import TensorProto, helper, numpy_helper

TASK="task301"; HERE=Path(__file__).resolve(); TASK_DIR=HERE.parents[1]; REPO=HERE.parents[4]
sys.path.insert(0,str(REPO/"workplace C"/"neurogolf-2026-work"/"scripts"))
from c_score_common import CURRENT_BEST_ONNX_DIR, score_onnx  # noqa:E402

def ini(n,v): return numpy_helper.from_array(np.asarray(v),name=n)

def build(path:Path)->Path:
    path.parent.mkdir(parents=True,exist_ok=True)
    ns=[
      helper.make_node("ReduceSum",["input","axes_hw"],["counts_f"],keepdims=1),
      helper.make_node("Cast",["counts_f"],["counts"],to=TensorProto.UINT8),
      helper.make_node("Slice",["counts","one","ten","axis_c"],["nonbg"]),
      helper.make_node("ReduceMax",["nonbg","axis_c"],["width"],keepdims=1),
      helper.make_node("ReduceSum",["counts_f","axis_c"],["area_f"],keepdims=1),
      helper.make_node("Cast",["area_f"],["area"],to=TensorProto.UINT8),
      helper.make_node("Div",["area","width"],["height"]), helper.make_node("Sub",["height","width"],["gap"]),
      helper.make_node("Add",["gap","nonbg"],["target_row"]), helper.make_node("Equal",["row1","target_row"],["row_mask"]),
      helper.make_node("Sub",["width","nonbg"],["left_edge"]), helper.make_node("Greater",["col1","left_edge"],["col_mask"]),
      helper.make_node("LessOrEqual",["row1","height"],["valid_rows"]), helper.make_node("LessOrEqual",["col1","width"],["valid_cols"]),
      helper.make_node("And",["row_mask","col_mask"],["bar0"]), helper.make_node("And",["bar0","valid_rows"],["bar1"]),
      helper.make_node("And",["bar1","valid_cols"],["bars"]), helper.make_node("Cast",["bars"],["bars_u8"],to=TensorProto.UINT8),
      helper.make_node("ReduceMax",["bars_u8","axis_c"],["occupied_u8"],keepdims=1), helper.make_node("Equal",["occupied_u8","zero_u8"],["empty"]),
      helper.make_node("And",["valid_rows","valid_cols"],["valid_canvas"]), helper.make_node("And",["empty","valid_canvas"],["background"]),
      helper.make_node("Concat",["background","bars"],["output"],axis=1),
    ]
    ins=[ini("axes_hw",np.array([2,3],np.int64)),ini("axis_c",np.array([1],np.int64)),ini("one",np.array([1],np.int64)),ini("ten",np.array([10],np.int64)),
         ini("row1",np.arange(1,31,dtype=np.uint8).reshape(1,1,30,1)),ini("col1",np.arange(1,31,dtype=np.uint8).reshape(1,1,1,30)),ini("zero_u8",np.array(0,np.uint8))]
    g=helper.make_graph(ns,"task301_direct_bar_masks",[helper.make_tensor_value_info("input",TensorProto.FLOAT,[1,10,30,30])],[helper.make_tensor_value_info("output",TensorProto.BOOL,[1,10,30,30])],ins)
    m=helper.make_model(g,opset_imports=[helper.make_opsetid("",18)],ir_version=8);onnx.checker.check_model(m);onnx.save(m,path);return path

def main():
    p=build(TASK_DIR/"onnx"/f"{TASK}_candidate.onnx"); old=score_onnx(TASK,CURRENT_BEST_ONNX_DIR/f"{TASK}.onnx");new=score_onnx(TASK,p)
    (TASK_DIR/"reports").mkdir(parents=True,exist_ok=True)
    with (TASK_DIR/"reports"/"cost_diff.csv").open("w",newline="",encoding="utf-8") as f:
      w=csv.DictWriter(f,fieldnames=["task","variant","passed","checked","cost","points","valid","artifact"]);w.writeheader()
      for v,r in [("baseline",old),("direct_bar_masks",new)]:w.writerow({"task":TASK,"variant":v,"passed":r.examples_passed,"checked":r.examples_checked,"cost":r.cost,"points":r.points,"valid":r.ok,"artifact":r.path})
    print(old);print(new)
if __name__=="__main__":main()
