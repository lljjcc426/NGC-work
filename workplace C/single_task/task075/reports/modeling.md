# task075 independent modeling

The 3x3 colored template is stored in the input's upper-left corner. Gray
column 3 separates it from a 3x3 marker lattice. Every marker cell with color
1 requests one copy of the template in the corresponding output block.

The alternative model extracts the 3x3 template with a spatial `Slice` and
decodes its one-hot channel with `ArgMax`. This replaces the parent's four-input
`Einsum` and removes `color_weights`, `sel3`, and `one_k`.

The rule is exact on all 265 examples. Its 90-cell one-hot slice costs more
memory than the parent's direct nine-scalar contraction, so it is rejected.

## Rank-one crop-convolution rewrite

The baseline template contraction is already exact rank one:

```text
T[c,h,w,p,q] = color_code[c] * select[p,h] * select[q,w]
```

Therefore the task104 rank-two sign construction cannot lower its algebraic
rank. More importantly, task075 must preserve ten exact colors rather than a
complementary two-color sign decision. The ten-color identity has real rank
10, so replacing the scalar color code with a two-sign code would require a
larger decoder and intermediate activations.

The accepted rewrite exploits the existing rank-one structure directly. A
`1x1 Conv` with the ten scalar color coefficients and pads
`[0,0,-27,-27]` performs the color contraction while cropping the input to the
top-left `3x3` template. This removes both `3x30` spatial selectors and the
singleton Einsum factor. A second `1x1 Conv`, cropped to rows `1,4,7` and
columns `5,8,11` with stride three, replaces the marker Slice and its four
control tensors.

Both convolutions produce exactly the same small tensors as the baseline, so
memory remains `1326`; initializer parameters fall from `161` to `68`.

- Official validation: `265/265`
- Cost: `1487 -> 1394`
- Points: `17.6954840535 -> 17.7600674087`
- Accepted: yes
