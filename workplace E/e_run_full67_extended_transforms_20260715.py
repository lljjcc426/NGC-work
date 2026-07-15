from __future__ import annotations

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO / "workplace C" / "neurogolf-2026-work" / "scripts"

EXTRA_TRANSFORMS = {
    "bool_reducesum": ("narrow_bool_reducesum.py", "narrow_bool_reducesum"),
    "bool_topk": ("narrow_bool_topk.py", "narrow_bool_topk"),
    "bool_extrema": (
        "narrow_bool_extrema_family.py",
        "narrow_bool_extrema_family",
    ),
    "bool_arg_extrema": (
        "narrow_bool_arg_extrema.py",
        "narrow_bool_arg_extrema",
    ),
    "float_add_sum": ("flatten_float_add_sum.py", "flatten_float_add_sum"),
    "integer_add_sum": (
        "flatten_integer_add_sum.py",
        "flatten_integer_add_sum",
    ),
    "integer_affine_constants": (
        "fold_integer_affine_constants.py",
        "fold_integer_affine_constants",
    ),
    "mul_global_sum": ("fold_mul_global_sum.py", "fold_mul_global_sum"),
    "variadic_minmax": (
        "flatten_variadic_minmax.py",
        "flatten_variadic_minmax",
    ),
    "nested_concat": ("flatten_nested_concat.py", "flatten_nested_concat"),
    "constant_concat_pad": (
        "concat_constant_to_pad.py",
        "concat_constant_to_pad",
    ),
}


if __name__ == "__main__":
    sys.path.insert(0, str(SCRIPT_DIR))
    import exact_transform_family_scan as scan

    scan.TRANSFORMS.update(EXTRA_TRANSFORMS)
    scan.main()
