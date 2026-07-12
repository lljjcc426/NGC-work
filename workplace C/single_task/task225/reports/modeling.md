# task225 independent modeling

Four colored source cells around an anchor define a repeating 6x6 pattern.
The alternative replaces the source-color 1x1 Conv with an explicit color
contraction `Einsum`, then reuses the parent's anchor and lookup construction.

It passes 265/265 and is cost-neutral at 1031, so no replacement is accepted.
