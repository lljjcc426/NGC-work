from __future__ import annotations

import argparse
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


TASK = "task055"
HERE = Path(__file__).resolve(); TASK_DIR = HERE.parents[1]; REPO = HERE.parents[4]
COMMON = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"
DEFAULT_PARENT = Path(r"E:/kagglegolf/submissions/candidates/GOLF_20260713_ALL399_DIRECT_13/onnx/task055.onnx")


def build(source: Path, output: Path) -> Path:
    model = deepcopy(onnx.load(str(source))); rebuilt = []
    for node in model.graph.node:
        if node.op_type == "Cast" and node.output[0] in {"rlf", "clf"}:
            next(a for a in node.attribute if a.name == "to").i = TensorProto.UINT8
            node.output[0] = "rlu" if node.output[0] == "rlf" else "clu"; rebuilt.append(node)
        elif node.op_type == "CumSum" and node.output[0] in {"rband", "cband"}:
            source_name = "rlu" if node.output[0] == "rband" else "clu"
            rebuilt.append(helper.make_node("QLinearConv", [source_name, "task055_qscale", "task055_qzero", "task055_prefix_kernel", "task055_qscale", "task055_qzero", "task055_qscale", "task055_qzero"], list(node.output), pads=[30, -1]))
        else: rebuilt.append(node)
    del model.graph.node[:]; model.graph.node.extend(rebuilt)
    kept=[x for x in model.graph.initializer if x.name not in {"arange3"}]
    del model.graph.initializer[:]; model.graph.initializer.extend(kept)
    model.graph.initializer.extend([
        numpy_helper.from_array(np.array([[[0],[1],[2]]],dtype=np.uint8),"arange3"),
        numpy_helper.from_array(np.asarray(1.0,dtype=np.float32),"task055_qscale"),
        numpy_helper.from_array(np.asarray(0,dtype=np.uint8),"task055_qzero"),
        numpy_helper.from_array(np.ones((1,1,30),dtype=np.uint8),"task055_prefix_kernel")])
    used={x for n in model.graph.node for x in n.input}; kept=[x for x in model.graph.initializer if x.name in used]
    del model.graph.initializer[:]; model.graph.initializer.extend(kept); del model.graph.value_info[:]
    output.parent.mkdir(parents=True,exist_ok=True); onnx.checker.check_model(model,full_check=True); onnx.shape_inference.infer_shapes(model,strict_mode=True); onnx.save(model,str(output)); return output


def main() -> None:
    ap=argparse.ArgumentParser();ap.add_argument("--parent",type=Path,default=DEFAULT_PARENT);a=ap.parse_args();sys.path.insert(0,str(COMMON));from c_score_common import score_onnx
    c=build(a.parent,TASK_DIR/"debug"/"task055_qconv_prefix.onnx");p=score_onnx(TASK,a.parent,True);r=score_onnx(TASK,c,True)
    print(json.dumps({"task":TASK,"valid":r.ok,"passed":r.examples_passed,"checked":r.examples_checked,"parent_cost":p.cost,"candidate_cost":r.cost,"delta_cost":None if r.cost is None else p.cost-r.cost,"sha256":r.sha256,"error":r.error}))
    if r.ok and r.cost is not None and r.cost<p.cost:
        out=TASK_DIR/"onnx"/"task055_candidate.onnx";out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(c,out);print(json.dumps({"accepted":str(out),"cost":r.cost}))


if __name__=="__main__":main()
