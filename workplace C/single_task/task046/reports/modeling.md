# task046 independent modeling

The task extracts colored path columns, fills gray gaps from neighboring
segments, and shifts three component rows according to path offsets.

The alternative decodes the complete 3x20 path region with spatial `Slice`
plus color-weight `Einsum`, replacing the cropped 1x1 Conv decoder. It passes
267/267 but materializes a 10-channel patch and is more expensive.
