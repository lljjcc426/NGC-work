from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def build(source: Path, output: Path) -> Path:
    model = onnx.load(source)
    initializers = {item.name: item for item in model.graph.initializer}
    spatial = numpy_helper.to_array(initializers["spatial_mix"])
    channel = numpy_helper.to_array(initializers["channel_mix"])
    if spatial.shape != (2, 2, 2) or channel.shape != (2, 10, 10):
        raise RuntimeError(f"unexpected shapes: spatial={spatial.shape}, channel={channel.shape}")
    if not np.array_equal(spatial[..., 1], np.asarray(-0.5, dtype=spatial.dtype) * spatial[..., 0]):
        raise RuntimeError("spatial mix does not have the required exact rank-one relation")

    spatial_factored = spatial[..., :1].copy()
    channel_factored = (channel[0] - np.asarray(0.5, dtype=channel.dtype) * channel[1])[None, ...]
    original = np.einsum("acv,vks->acks", spatial, channel)
    factored = np.einsum("acv,vks->acks", spatial_factored, channel_factored)
    if not np.array_equal(original, factored):
        raise RuntimeError("factored terminal mix is not elementwise exact")

    replacements = {
        "spatial_mix": numpy_helper.from_array(spatial_factored, "spatial_mix"),
        "channel_mix": numpy_helper.from_array(channel_factored, "channel_mix"),
    }
    updated = [replacements.get(item.name, item) for item in model.graph.initializer]
    del model.graph.initializer[:]
    model.graph.initializer.extend(updated)
    del model.graph.value_info[:]

    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(inferred, output)
    reloaded = onnx.load(output)
    onnx.checker.check_model(reloaded, full_check=True)
    onnx.shape_inference.infer_shapes(reloaded, strict_mode=True, data_prop=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Factor task224's terminal spatial/channel mix axis exactly.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(build(args.source, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
