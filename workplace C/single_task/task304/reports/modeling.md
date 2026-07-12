# task304 independent modeling

The top-left 3x3 grid contains colors 1..9. The most frequent color is used as
a mask: matching grid cells reproduce the full 3x3 pattern in a 9x9 output.

The alternative model decodes the 3x3 scalar grid first, builds a nine-bin
one-hot histogram with `Equal`, and selects the majority using `ReduceSum` and
`ArgMax`. This removes the parent's independent count `Einsum`.

All 266 examples pass. The explicit 9x3x3 histogram costs more memory than the
parent's nine-scalar contraction, so it is rejected.
