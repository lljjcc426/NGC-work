# FINAL TASK315 REPORT

- Rule understood: place a copy of the input pattern in every output tile selected by an input color-2 cell.
- Independent structure: `Slice/Gather -> six-dimensional broadcast Where -> Transpose -> Reshape`.
- Full validation: 266/266 exact.
- Baseline cost: 230.
- Candidate cost: 3252.
- Delta cost: +3022.
- Candidate: `onnx/task315_kronecker.onnx`.
- Accepted replacement: no.
- Result: the explicit Kronecker model is exact, but materialized block tensors outweigh the fused Einsum.

## Rank-Two Sign Probe

The B/task104 sign-factor method was tested on the baseline single Einsum. A
two-dimensional color factor reduces nominal cost from 230 to 210 and is exact
under argmax, but NeuroGolf decodes every output channel independently with
`output > 0`. The candidate emits additional positive channels and therefore
passes `0/266` official examples. Position-only and full color/position rank-2
searches reached at most `23/266` under argmax. The rank-two rewrite is
rejected; future searches must optimize per-channel sign margin directly.
