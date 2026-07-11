# FINAL TASK278 REPORT

Generated: 2026-07-11

## Attempt

The dedicated candidate convolved the 3x3 anchor kernel with a 3x3 all-ones
kernel and replaced `QLinearConv -> MaxPool` with one 5x5 QLinearConv.

## Result

- Baseline cost: `4503`
- Fused candidate cost: `4195`
- Nominal delta cost: `308`
- Full validation: `0/265`
- Status: rejected

A sweep of output quantization scales from 1 through 128 produced no fully
correct example. The original operation thresholds local anchor evidence first
and then ORs neighboring decisions. That two-stage nonlinearity is not
preserved by a single linear 5x5 threshold.

## Decision

Do not submit or merge this artifact. Revisit task278 only with a representation
that retains separate anchor detection and halo dilation while reducing another
part of the graph.
