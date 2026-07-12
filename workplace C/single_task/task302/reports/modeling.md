# task302 independent modeling

The task recognizes a color-5 frame, fills its interior, and emits fixed
colors for frame/interior geometry. The alternative replaces the first two
quantized detectors with float Conv, an explicit frame threshold, and float
fill dilation.

The model is structurally valid but fails 266/266 outputs: quantized rounding
and saturation are part of the rule and are not reproduced by the naive float
threshold. The failed candidate is retained as a deep model attempt.
