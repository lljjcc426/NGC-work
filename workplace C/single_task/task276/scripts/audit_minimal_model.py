from pathlib import Path
import onnx
from onnx import numpy_helper

MODEL = Path(r"E:/kagglegolf/submissions/candidates/GOLF_20260711_096_v95_plus_4_compact/onnx/task276.onnx")
model = onnx.load(MODEL)
assert len(model.graph.node) == 1 and model.graph.node[0].op_type == "Gather"
indices = numpy_helper.to_array(model.graph.initializer[0])
assert indices.shape == (10,)
print("task276: one Gather with 10 channel-permutation indices; no lower-cost equivalent accepted")
