# task335 independent modeling

Two colored markers define a rectangle and output border/fill basis masks. The
parent contracts a sparse 4x4 template directly with row and column bases.

The alternative explicitly forms all 16 row/column outer-product bases,
flattens them, and applies the template with MatMul. It passes 266/266 but costs
77146 due to the 4x4x30x30 basis tensor.
