# task286 independent modeling

The task identifies a seeded passable component and recolors its connected
region while preserving barriers. The Python rule solver passes all examples.

The alternative ONNX replaces seven packed `MatMulInteger` operations with
float Cast, MatMul, and int32 Cast paths while preserving the complete bitset
propagation graph. It passes 265/265 but costs 45209 versus 26909.
