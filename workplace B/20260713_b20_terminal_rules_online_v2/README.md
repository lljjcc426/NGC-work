# B terminal-rule integration: 7290.38

This folder continues from the accepted `7282.01` B package and changes only
four B-owned tasks. Every task was submitted incrementally, so its online gain
is isolated rather than inferred from a blend.

| Task | Cost | Points | Local gain | Kaggle ref | Online score |
| --- | ---: | ---: | ---: | ---: | ---: |
| task161 | 275 | 19.383229 | +1.867036 | 54614776 | 7283.88 |
| task163 | 310 | 19.263428 | +1.753957 | 54615076 | 7285.64 |
| task212 | 412 | 18.978977 | +1.697218 | 54615262 | 7287.33 |
| task350 | 428 | 18.940877 | +3.049849 | 54615459 | 7290.38 |

The four tasks add `+8.37` online over the `7282.01` baseline. The final zip
contains 400 models and has SHA-256
`FBF70D2171E37613C40A727E4EF230F7FA5FFCB6BB9E1CDB41A3E95251BA926D`.

## Structural findings

- task161 detects the unique four-endpoint color and renders its horizontal and
  vertical cross with a compact terminal contraction.
- task163 compiles the 3x3 tile-routing rule directly into one Einsum. The
  marker's position inside its source tile chooses the destination tile.
- task212 compiles the separator-aware ray extension into one zero-memory
  terminal Einsum.
- task350 compiles horizontal and vertical first-to-last point spans into one
  zero-memory terminal Einsum. It passed all 267 supplied examples plus 500
  independently generated binary grids spanning dimensions 1 through 30.

## Rejected experiment

The task163 constants contain only 88 nonzero values despite 310 dense
parameters. Converting them to ONNX sparse initializers would score above 20 in
theory, but the official strict shape inference reports sparse Einsum inputs as
rank zero. The model is therefore invalid before execution. The reproduction
script is retained under `scripts/`; do not submit its output.

