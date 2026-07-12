# task392 independent modeling

The input encodes a spiral's color and geometry; the output reconstructs the
10x10 spiral against gray. The alternative decodes the unique non-background
color from a channel-presence vector and `ArgMax`, replacing scalar
color-sum/count division while preserving the parent's geometry decoder.

It passes 266/266 but costs more than the scalar color contractions.
