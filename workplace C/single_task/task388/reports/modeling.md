# task388 independent modeling

The task builds an N x N separator-augmented base pattern and tiles that base
twice along both axes to produce a 2N x 2N output.

The alternative dynamically slices the valid N x N base and uses one `Tile`
instead of two fixed-index Gather stages. It passes 266/266. The official
profiler does not return memory for this dynamic Tile graph, so no accepted
cost claim is made.
