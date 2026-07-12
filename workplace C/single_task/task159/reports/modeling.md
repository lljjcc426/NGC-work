# task159 independent modeling

The input contains a colored border rectangle and a separate five-cell motif.
The output crops the rectangle and tiles the motif according to rectangle size.
The motif's minimum row and column select the source patch.

The alternative model computes motif occupancy for all 30 rows and columns,
uses `Sign` plus `ArgMax` to locate the first occupied axis coordinate, and
reuses the parent's patch and repeat path. This replaces logarithmic base-4
coordinate encoding with explicit geometric localization.

All 265 examples pass. Two 30-element occupancy vectors cost more than the
parent's scalar exponent encoding, so the candidate is rejected.
