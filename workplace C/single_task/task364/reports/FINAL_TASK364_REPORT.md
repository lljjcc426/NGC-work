# FINAL TASK364 REPORT

Generated: 2026-07-11

## Attempt

The baseline performs five masked 3x3 propagation rounds after constructing a
component seed. The dedicated probe bypasses rounds four and five and feeds the
third-round state directly to the output classifier.

## Result

- Baseline public validation: `266/266`
- Three-round probe validation: `199/266`
- Baseline official cost: `14642`
- Probe reported cost: `14642`
- Status: rejected

The 67 failures show that the last two rounds are semantically required for
larger public components. Larger one-shot pooling kernels were also rejected
because they cross barriers that the repeated `MaxPool -> Mul(mask)` sequence
preserves.

## Next Viable Direction

Do not reduce propagation depth again. A future task364 candidate must replace
the full propagation mechanism with a component signature classifier or an
equivalent barrier-preserving representation.

## Accepted Orthogonal Compression

A separate opset-18 compact Pad rewrite preserves the five-round baseline:

- validation: `266/266`
- parent cost: `14642`
- candidate cost: `14640`
- delta points: `+0.00013660269128479285`

This candidate is local accepted and has not been submitted.

## Fused Seed Expansion Probe

A dedicated 5x5 seed-expansion fusion reduced nominal cost from 14642 to
12882, but passed only 110/266 examples. Separate A-only and B-only fusions
also failed (157/266 and 161/266). The two seed branches encode distinct
component boundary conditions and cannot be replaced by the tested shared
kernel. This route is rejected.
