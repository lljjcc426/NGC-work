# task276 independent modeling

The task swaps colors 2 and 6 while preserving every other channel. The
alternative reconstructs the channel permutation with ten channel Slice nodes
and one Concat rather than a single Gather.

It passes 266/266 but costs 36031 versus 10, demonstrating why direct channel
Gather is the correct compact representation.
