# task077 Independent Propagation Modeling

## Rule and parent structure

Color 2 supplies the shape mask. The model builds a vertical admissibility
mask, then performs three rounds of barrier-constrained horizontal propagation.
A final vertical bridge fills a cell when the propagated center is present or
both vertical neighbors are present. Filled non-source cells become color 4.

The effective parent cost is 7655 and passes all 266 public train/test/arc-gen
examples.

## Propagation experiments

### Ordinary two-round schedules

Removing the third `MaxPool -> Min(mask)` pair lowers cost to 6817, but the best
ordinary 5/5 schedule passes only 258/266. The nine output errors are at the
ends of length-seven admissible segments: two radius-two rounds cannot reach
them.

### Dilated two-round schedules

`scripts/search_dilated_two_round.py` tested 55 legal combinations using odd
kernels 3..13 and dilation 1 or 2. The best schedule passes only 227/266.
Sparse jumps cross short barriers before the per-round mask can reject them, so
dilation cannot replace the third constrained round.

### Single-step constrained morphology

The required update is `mask AND horizontal_OR(state)`. A single convolution
cannot consume both dynamic tensors without first materializing a two-channel
concatenation, which costs more than the eliminated MaxPool tensor. Direct
large-kernel morphology was rejected by the full schedule tests because it can
jump disconnected admissible components.

## Accepted quantized fusion

The parent vertical bridge uses quantized kernel `[1,3,1]` with output scale 3,
so its response `S` is 0, 1, or 2. It materializes a second tensor `T=2R` and
tests `S>T` to suppress original color-2 cells.

The candidate changes the bridge kernel to `[1,2,1]`. Every positive legal
pattern has accumulator 2..4 and quantizes to exactly 1. Therefore the original
predicate is exactly equivalent to `S>R`. This removes the complete `T` tensor
and its 1x1 quantized-convolution weight while preserving all three safe
propagation rounds.

## Official result

- Parent: 266/266, memory 7620, params 35, cost 7655.
- Candidate: 266/266, memory 7200, params 34, cost 7234.
- Delta: -421 cost, +0.056566895579742 points.
- Status: locally accepted.
