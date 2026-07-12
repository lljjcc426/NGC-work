# task077 independent modeling

The model propagates horizontal fill through a vertical barrier mask. The
parent performs three repeated `MaxPool -> Min(mask)` rounds.

The dedicated candidate removes one complete propagation round and feeds the
second-round state into the vertical classifier. A kernel schedule sweep found
the 5/5 two-round variant to be best: it lowers nominal cost from 7657 to 6817
but passes only 258/266 examples. Larger spans cannot replace the missing
barrier-preserving iteration.

This structural candidate, not the unrelated two-cost Pad rewrite, is the
basis for marking task077 independently modeled.
