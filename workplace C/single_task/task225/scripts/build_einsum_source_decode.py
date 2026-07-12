from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import onnx
from onnx import helper

def build(source: Path, output: Path) -> Path:
    model=deepcopy(onnx.load(source));node=model.graph.node[16]
    if node.op_type!="Conv" or node.output!=["source_code_f"]:raise RuntimeError("unexpected task225 graph")
    replacement=helper.make_node("Einsum",["source_block","source_kernel"],["source_code_f"],equation="bchw,ockl->bohw",name="decode_source_colors")
    old=list(model.graph.node);del model.graph.node[:];model.graph.node.extend(old[:16]+[replacement]+old[17:]);output.parent.mkdir(parents=True,exist_ok=True);onnx.checker.check_model(model,full_check=True);onnx.save(model,output);return output
def main():
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();print(build(a.source,a.output))
if __name__=="__main__":main()
