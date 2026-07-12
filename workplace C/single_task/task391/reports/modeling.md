# task391 independent modeling

The output lists three selected low-frequency non-background colors. The
alternative computes the color histogram in two stages: spatial ReduceSum then
batch squeeze, before reusing the TopK selection path.

It passes 267/267 but costs 202 versus 159 because the staged histogram adds an
intermediate vector.
