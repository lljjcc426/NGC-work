from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from task_model import build_onnx
if __name__ == "__main__":
    print(build_onnx(Path(__file__).resolve().parents[1] / "onnx/task332_candidate.onnx"))
