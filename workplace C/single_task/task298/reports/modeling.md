# task298 independent modeling

The right-hand color strip defines a cyclic color substitution. The parent
first materializes the three strip cells and then applies the substitution to
the complete input.

The dedicated replacement folds both strip selections into the final
`Einsum`. Two shared selector matrices pick rows 0..2 and column 2 directly
from the original one-hot input, while the existing 3x3 rotation matrix maps
source colors to destination colors. No intermediate tensor is produced.

This rule is exact on all 267 public examples and lowers official cost.
