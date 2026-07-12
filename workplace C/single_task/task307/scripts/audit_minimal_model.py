from pathlib import Path
import math
import onnx

MODEL = Path(r"E:/kagglegolf/submissions/candidates/GOLF_20260711_096_v95_plus_4_compact/onnx/task307.onnx")
model = onnx.load(MODEL)
assert len(model.graph.node) == 1 and model.graph.node[0].op_type == "MaxRoiPool"
assert sum(math.prod(init.dims) for init in model.graph.initializer) == 5
print("task307: one MaxRoiPool, ROI tensor has 5 parameters; no lower-cost equivalent accepted")
