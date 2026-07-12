# task349 independent modeling

The Python model detects color-9 rectangles, emits a downward ray, and builds a
width-dependent halo. Two-channel and four-channel packed halo encodings were
constructed and tested against all public neighborhoods; overlapping small
rectangles produce false positives, so those aggressive structures were
rejected.

The retained candidate shortens the five-channel detector after proving width
10 is the public maximum. It passes 267/267 and lowers cost by 5. The deeper
packed alternatives remain recorded as failed rule models rather than claimed
improvements.
