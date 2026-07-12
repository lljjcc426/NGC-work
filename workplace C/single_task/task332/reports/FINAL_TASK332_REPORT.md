# FINAL TASK332 REPORT

- Candidate: compact row-code Conv.
- Full validation: `267/267`.
- Parent/candidate cost: `561 -> 438`.
- Points: `18.6702790945 -> 18.9177810896`.
- Delta points: `+0.2475019951`.
- Decision: local accepted.

The accepted model replaces `Einsum -> Slice`, which first materialized a
30-column row-code tensor, with one 3x1 Conv. Its negative bottom/right pads
directly produce the required 1x1x1x20 tensor. The dynamic width-parity logic
and final quantized classifier remain unchanged.
