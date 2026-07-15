from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import build_blend  # noqa: E402
import optimize_equivalent as oe  # noqa: E402
from rewrite_task285_unpad_flood_current import official_equivalence, stress_test  # noqa: E402


DEFAULT_BASE = (
    ROOT
    / "public_probe_variants"
    / "team_submission5_b_work_20260716"
    / "submission"
    / "task285.onnx"
)


def rewrite(source: onnx.ModelProto, diagonal_offset: int) -> onnx.ModelProto:
    model = onnx.ModelProto.FromString(source.SerializeToString())
    if diagonal_offset == 0:
        matches = [node for node in model.graph.node if "exact_anchor_score_0" in node.output]
        if len(matches) != 1 or matches[0].input[0] != "exact_has_same":
            raise RuntimeError("task285 anchor score node was not found")
        matches[0].input[0] = "task285_has_same4"
    else:
        replacement = numpy_helper.from_array(
            np.array([[diagonal_offset]], dtype=np.int32),
            "diagonal_offsets",
        )
        for index, initializer in enumerate(model.graph.initializer):
            if initializer.name == "diagonal_offsets":
                model.graph.initializer[index].CopyFrom(replacement)
                break
        else:
            raise RuntimeError("task285 diagonal_offsets initializer was not found")

    del model.graph.value_info[:]
    oe.prune_dead(model)
    oe.prune_initializers(model)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stress", type=int, default=0)
    parser.add_argument("--seed", type=int, default=285_20260716)
    parser.add_argument("--diagonal-offset", type=int, choices=[-31, 0, 31], default=0)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(rewrite(onnx.load(args.base), args.diagonal_offset), args.out)
    result = {
        "task": 285,
        "method": f"source anchor diagonal offset={args.diagonal_offset}",
        "equivalence": official_equivalence(args.base, args.out),
        "stress": stress_test(args.out, args.stress, args.seed) if args.stress else None,
        "score": build_blend.validate_and_score((285, "task285_anchor_orth", str(args.out))),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
