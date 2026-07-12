# task373 independent modeling

## Rule

The two input rows are uniform and provide colors `A` and `B`. The output is a 2x6 checkerboard:

```text
A B A B A B
B A B A B A
```

No arithmetic relation between color IDs is used; the colors are copied from the first cell of each input row.

## Independent structure

`scripts/build_checkerboard_concat.py` slices the two source color cells, creates the two alternating rows with task-specific Concat nodes, concatenates the rows, and pads the 2x6 checkerboard. It fully replaces the signed-vector Einsum used by the parent model.

## Official validation and cost

| variant | passed | checked | memory | params | cost |
|---|---:|---:|---:|---:|---:|
| signed Einsum | 75 | 75 | 0 | 60 | 60 |
| Slice/Concat checkerboard | 75 | 75 | 1040 | 18 | 1058 |

The explicit checkerboard is exact and reduces constants from 60 to 18 elements, but intermediate rows and concatenations cost more than the fused Einsum.
