# task061 independent modeling

The output is a modular multiplication table whose modulus is decoded from the
input palette. The alternative casts row and column residues to float, forms
their outer product with ordinary `MatMul`, casts back to uint8, and applies
the existing modulo/output path.

It passes 267/267 but costs more than the quantized matrix multiplication.
