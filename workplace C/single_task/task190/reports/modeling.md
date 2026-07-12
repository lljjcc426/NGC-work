# task190 independent modeling

The task recognizes a nearly complete 2x2 corner plus a diagonal endpoint and
extends the endpoint as a diagonal ray in its inferred direction.

The alternative replaces all four quantized ray convolutions with explicit
float Cast, Conv, and Cast paths while retaining the same four directional
detectors. It passes 266/266, proving the directional rule independently, but
the added spatial tensors increase cost.
