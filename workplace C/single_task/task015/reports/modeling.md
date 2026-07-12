# task015 independent modeling

The task expands sparse colored marker relations into colors 4 and 7. The
parent encodes the complete local rule in one sparse 10x10 convolution.

The alternative splits input channels into two groups, evaluates two 3x3
convolutions, and adds their outputs. It passes 265/265 but materializes two
full output tensors and costs 109805 versus 900.
