# task278 independent modeling

The parent thresholds 3x3 anchor evidence and then dilates accepted anchors.
The independent model convolves the anchor kernel with a 3x3 all-ones kernel
and replaces `QLinearConv -> MaxPool` with one 5x5 quantized convolution.

Nominal cost falls from 4503 to 4195, but all 265 examples fail across output
scale sweeps. A single linear threshold cannot reproduce threshold-then-OR
nonlinearity.
