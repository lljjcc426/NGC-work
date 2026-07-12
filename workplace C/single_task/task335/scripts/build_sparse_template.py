from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, numpy_helper

def build(source: Path, output: Path) -> Path:
    model=deepcopy(onnx.load(source));dense=next(x for x in model.graph.initializer if x.name=="T");array=numpy_helper.to_array(dense)
    coords=np.argwhere(array!=0).astype(np.int64);values=array[tuple(coords.T)]
    sparse=helper.make_sparse_tensor(numpy_helper.from_array(values,name="T"),numpy_helper.from_array(coords,name="T_indices"),list(array.shape))
    kept=[x for x in model.graph.initializer if x.name!="T"];del model.graph.initializer[:];model.graph.initializer.extend(kept);model.graph.sparse_initializer.append(sparse)
    output.parent.mkdir(parents=True,exist_ok=True);onnx.checker.check_model(model,full_check=True);onnx.save(model,output);return output
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();print(build(a.source,a.output))
if __name__=="__main__":main()
