# task141 independent modeling

The input contains one colored seed in an N x N grid. The output draws both
diagonals through that seed, preserving its color.

The alternative model decodes the complete scalar grid, flattens it, locates
the unique seed with `ArgMax`, derives row and column by integer division and
modulo, and reuses the parent's diagonal-distance mask.

All 265 examples pass. Materializing and flattening the 30x30 scalar grid is
far more expensive than the parent's three scalar contractions, so the
candidate is rejected.
