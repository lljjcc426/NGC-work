# task347 independent modeling

The output is the union of the color-4 mask in the left 3x3 panel and the
color-3 mask in the right panel, recolored to 6.

The alternative model explicitly slices color 4 and color 3, computes their
union with `Max`, derives the background channel by subtraction, concatenates
the two channels, and reuses the sparse unpool placement.

It passes all 269 examples. Explicit arithmetic materializes three additional
3x3 tensors, so the parent's comparison-plus-template formulation is cheaper.
