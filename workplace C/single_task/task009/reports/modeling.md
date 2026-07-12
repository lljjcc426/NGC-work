# task009 independent modeling

The task decodes a coarse sampled grid and separates true background cells
from outside padding. The independent model encodes that distinction directly
inside the first convolution: valid colors decode to 0..9 and all-zero padded
input decodes to sentinel 10.

This deletes a complete 10x10 validity-mask and `Where` path. It passes all 266
examples and lowers official cost from 6694 to 6595. Compacting the two
single-axis Pad controls then lowers the best cost again to 6585 with 265/265
full validation.
