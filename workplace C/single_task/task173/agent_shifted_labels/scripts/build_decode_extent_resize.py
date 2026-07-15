from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

from build_decode_extent import (
    AGENT_DIR,
    COMMON,
    DEFAULT_PARENT,
    DEFAULT_SOURCE,
    build as build_decode_extent,
)


DEFAULT_OUTPUT = AGENT_DIR / "debug" / "task173_candidate.onnx"
DEFAULT_EVIDENCE = AGENT_DIR / "debug" / "candidate_evidence.json"


def build(source: Path, output: Path) -> Path:
    build_decode_extent(source, output)
    model = onnx.load(output)

    removed = {"supd1", "supd2", "supdf"}
    kept_nodes = []
    inserted = False
    for node in model.graph.node:
        if node.output and node.output[0] in removed:
            if node.output[0] == "supdf":
                kept_nodes.append(
                    helper.make_node(
                        "Resize",
                        ["supd0", "", "scale4"],
                        ["supdf"],
                        name="repeat_updates_four_times",
                        coordinate_transformation_mode="asymmetric",
                        mode="nearest",
                        nearest_mode="floor",
                    )
                )
                inserted = True
            continue
        kept_nodes.append(node)
    if not inserted:
        raise RuntimeError("update expansion chain was not found")
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)

    kept_initializers = [
        item for item in model.graph.initializer if item.name != "shM4"
    ]
    kept_initializers.append(
        numpy_helper.from_array(np.asarray([4.0], dtype=np.float32), name="scale4")
    )
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)
    del model.graph.value_info[:]

    model.graph.name = "task173_safe_decode_extent_resize"
    model.producer_name = "ngc-task173-safe-decode-extent-resize"
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    supdf = next(item for item in inferred.graph.value_info if item.name == "supdf")
    dims = [dim.dim_value for dim in supdf.type.tensor_type.shape.dim]
    if dims != [28]:
        raise RuntimeError(f"unexpected Resize output shape: {dims}")
    onnx.save(model, output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    args = parser.parse_args()

    sys.path.insert(0, str(COMMON))
    from c_score_common import score_onnx
    from candidate_registry import operator_audit

    candidate = build(args.source, args.output)
    parent_result = score_onnx("task173", args.parent, validate_all=True)
    source_result = score_onnx("task173", args.source, validate_all=True)
    candidate_result = score_onnx("task173", candidate, validate_all=True)
    audit = operator_audit(candidate)
    evidence = {
        "task": "task173",
        "rewrite": [
            "derive cropped row-column extents from fractional label Conv",
            "replace uint8 Unsqueeze-Expand-Reshape repetition with 1-D Resize",
        ],
        "parent": parent_result.__dict__,
        "safe_dilated_source": source_result.__dict__,
        "candidate": candidate_result.__dict__,
        "delta_from_parent": (
            None
            if parent_result.cost is None or candidate_result.cost is None
            else parent_result.cost - candidate_result.cost
        ),
        "delta_from_safe_dilated_source": (
            None
            if source_result.cost is None or candidate_result.cost is None
            else source_result.cost - candidate_result.cost
        ),
        "operator_audit": audit,
    }
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    return 0 if candidate_result.ok and audit["runtime_compatible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
