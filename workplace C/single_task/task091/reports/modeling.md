# task091 independent modeling

Color 5 forms two vertical guide segments. Their first and last occupied rows
and their left and right columns define the output crop. Color 8 is copied
inside that crop while the guide rows and columns reconstruct color 5.

The alternative model obtains the guide's upper and lower bounds from a direct
gray-channel row projection. It removes the left-column `ArgMax`, `Squeeze`,
and cast decoder and uses an independent row `Einsum` plus two `ArgMax` nodes.

All 266 examples pass. The 30-element row projection increases scored memory,
so that candidate is rejected. A second, exact metadata/control-path rewrite
compacts dynamic Pad axes and lowers official cost from 2759 to 2730 while
remaining 266/266; that candidate is accepted locally.
