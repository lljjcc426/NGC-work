# task381 independent modeling

The task fills background cells horizontally bounded by red components with
maroon. The alternative uses forward and reversed `CumSum` scans to detect a
red cell on both sides of every position.

It passes 265/265, independently confirming the span rule. The two explicit
scan tensors cost more than the parent's weighted MatMul encoding.
