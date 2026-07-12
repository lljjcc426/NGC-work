from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import onnx
from onnx import TensorProto, helper

def build(source: Path, output: Path) -> Path:
    model=deepcopy(onnx.load(source));old=list(model.graph.node)
    if old[5].op_type!="QLinearMatMul":raise RuntimeError("unexpected task061 graph")
    repl=[helper.make_node("Cast",["row_mod"],["row_f"],to=TensorProto.FLOAT),helper.make_node("Cast",["col_mod"],["col_f"],to=TensorProto.FLOAT),helper.make_node("MatMul",["row_f","col_f"],["product_f"]),helper.make_node("Cast",["product_f"],["product"],to=TensorProto.UINT8)]
    del model.graph.node[:];model.graph.node.extend(old[:5]+repl+old[6:])
    output.parent.mkdir(parents=True,exist_ok=True);onnx.checker.check_model(model,full_check=True);onnx.save(model,output);return output
def main():
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();print(build(a.source,a.output))
if __name__=="__main__":main()
