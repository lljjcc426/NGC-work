from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


TASK_DIR = Path(__file__).resolve().parents[1]
SOURCE = Path(
    r"E:/kagglegolf/submissions/candidates/"
    r"GOLF_20260711_096_v95_plus_4_compact/onnx/task077.onnx"
)
OUTPUT = TASK_DIR / "onnx" / "task077_candidate.onnx"


def replace_initializer(model: onnx.ModelProto, name: str, value: np.ndarray) -> None:
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name == name:
            model.graph.initializer[index].CopyFrom(numpy_helper.from_array(value, name=name))
            return
    raise KeyError(name)


def build(output_path: Path = OUTPUT) -> Path:
    """Fuse the original-pixel exclusion into the binary vertical response.

    With output scale 3, the `[1, 2, 1]` vertical kernel maps every positive
    neighborhood (sum 2..4) to uint8 one. Therefore `S > R` is true exactly on
    propagated non-source cells. The parent used `[1, 3, 1]`, produced values
    up to two, and needed a separate `T = 2*R` tensor before `S > T`.
    """
    model = onnx.load(str(SOURCE))
    replace_initializer(model, "wv", np.array([[[[1], [2], [1]]]], dtype=np.uint8))

    nodes = list(model.graph.node)
    threshold = next(node for node in nodes if node.output == ["T"])
    greater = next(node for node in nodes if node.output == ["fill"])
    greater.input[1] = "R"
    nodes.remove(threshold)
    del model.graph.node[:]
    model.graph.node.extend(nodes)

    kept = [initializer for initializer in model.graph.initializer if initializer.name != "w2"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(output_path))
    return output_path


if __name__ == "__main__":
    print(build())
