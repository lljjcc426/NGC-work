# task237 independent modeling

The task decodes packed row markers and extends their colors horizontally and
vertically. The parent uses fractional float convolution weights to pack color
and position in one value.

The alternative scales those weights by 16, uses exact `ConvInteger`, and
divides the integer result by 16 before the existing decoder. It passes all
266 examples, but the integer/float conversion tensors raise cost.

## 2026-07-12 compact packed Conv

The packed Conv previously emitted 30 rows and a following Slice retained the
first nine. Moving the crop into the Conv as a negative bottom pad produces the
nine-row tensor directly. Official validation passes 266/266 and cost falls
from 1836 to 1716.
