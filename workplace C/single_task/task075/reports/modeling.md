# task075 independent modeling

The 3x3 colored template is stored in the input's upper-left corner. Gray
column 3 separates it from a 3x3 marker lattice. Every marker cell with color
1 requests one copy of the template in the corresponding output block.

The alternative model extracts the 3x3 template with a spatial `Slice` and
decodes its one-hot channel with `ArgMax`. This replaces the parent's four-input
`Einsum` and removes `color_weights`, `sel3`, and `one_k`.

The rule is exact on all 265 examples. Its 90-cell one-hot slice costs more
memory than the parent's direct nine-scalar contraction, so it is rejected.
