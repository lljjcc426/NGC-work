# B task018 compact exact rewrite on the 7420.93 parent

This batch continues from Kaggle ref `54736568` at `7420.93` and changes only
the B-assigned `task018` model. It is below the one-point direct-submit
threshold and has not been submitted yet.

## Accepted task018 rewrite

The rewrite keeps the accepted sparse dihedral solver's behavior while reducing
four intermediate paths:

- Coordinate bounds use `coord == Clip(coord)` instead of separate lower and
  upper comparisons plus uint8 reduction.
- Flat 24x24 lookup indices use one `Einsum` with `[24, 1]` instead of two
  gathers, a multiply, and an add.
- Point-to-anchor grouping computes two distance vectors directly instead of a
  `[24, 2, 2]` distance tensor followed by `ArgMin`.
- `ScatterND` consumes native `[2, 12, 2]` indices and `[2, 12]` updates, so the
  two flattening `Reshape` nodes are removed.

Results:

- Cost: `17025 -> 15754`.
- Points: `15.278094277923 -> 15.335150419737`.
- Unpublished gain over the online parent: `+0.057056141814`.
- Official validation: `266/266` train, test, and ARC-GEN examples.
- Fresh differential validation: `18000/18000` on the algebraic core and a
  further `2000/2000` after the rank-3 `ScatterND` rewrite.
- Model SHA256:
  `26C109A847F83AA0757F41E607740E5D90C2001D7A5E5BA00AFEDCB9BDB36A70`.

The working package still contains all 400 tasks. About `+0.942944` additional
gain is required before the next direct Kaggle submission.

## Rejected task285 experiments

The lower-scoring `task285` was attacked first, but no candidate was accepted:

- Direct Pad removal passed official data but produced an out-of-range hidden
  generator index.
- Four flood steps failed fresh generator example 794.
- Five padded flood steps failed fresh generator example 6588.
- Orthogonal-only anchors failed official train example 0.
- Single diagonal offsets `-31` and `+31` failed official train examples 0 and
  2 respectively, proving that both legacy directions are required.

The rejected ONNX files are not included. Their reproducible search scripts are
kept under `scripts/` so these dead ends are not repeated.

## Contents

- `models/task018.onnx`: accepted override.
- `scripts/rewrite_task018_clip_bounds_current.py`: reproducible accepted build.
- `scripts/rewrite_task285_*.py`: documented rejected search paths.
- `reports/summary.json`: machine-readable score and validation summary.
