# task193 independent modeling

The task expands local rule evidence with a spatial convolution. The dedicated
model replaces the dense convolution with a depthwise rule implementation.
It is mathematically exact on all 266 examples but materializes large spatial
intermediates, increasing cost from 910 to 82912.
