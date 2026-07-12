# FINAL TASK015 REPORT

- Candidate: split-input sparse convolution plus output sum
- Validation: `265/265`
- Parent/candidate cost: `900 -> 109805`
- Decision: independently modeled; reject candidate

## 2026-07-12 grouped large-window search

Larger grouped Conv windows are structurally infeasible, not merely hard to
optimize: identical windows visible to a group have both positive and negative
labels. Group=5 4x4/5x5 conflicts on channels 0, 4, and 7; group=2 4x4
conflicts on channel 7. The dense single-Conv baseline remains preferred.
