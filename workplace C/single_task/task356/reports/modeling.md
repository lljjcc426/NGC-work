# task356 independent modeling

The task fills every horizontal or vertical span bounded by color-8 cells.
The alternative computes forward and reverse cumulative occupancy scans on
both axes, intersects each pair, and unions horizontal and vertical spans.

It passes 266/266. Four explicit scan tensors cost more than the parent's four
directional MaxPool operations.
