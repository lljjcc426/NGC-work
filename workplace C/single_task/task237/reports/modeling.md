# task237 independent modeling

The task decodes packed row markers and extends their colors horizontally and
vertically. The parent uses fractional float convolution weights to pack color
and position in one value.

The alternative scales those weights by 16, uses exact `ConvInteger`, and
divides the integer result by 16 before the existing decoder. It passes all
266 examples, but the integer/float conversion tensors raise cost.
