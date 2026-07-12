# task307 independent modeling

The task doubles every input cell into a 2x2 block. The alternative slices the
active 15x15 region and applies nearest-neighbor Resize with scale 2 instead of
MaxRoiPool.

It passes 266/266. The official profiler reports 9000 memory but no parameter
total for this Resize graph, so no accepted cost claim is made.
