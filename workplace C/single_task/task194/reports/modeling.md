# task194 independent modeling

## Rule

The logical input is 3x3. The 6x6 output is a rotational four-quadrant composition:

```text
[ input       rotate90(input)  ]
[ rotate270   rotate180(input) ]
```

The rule holds for all 266 public examples, including rotationally symmetric cases where multiple transforms happen to look equal.

## Structural candidate

`scripts/build_rotation_quadrants.py` crops the logical 3x3 input, creates rotations with `Transpose` and negative-step `Slice`, joins them with three `Concat` nodes, and pads the 6x6 result to the 30x30 benchmark tensor. This fully replaces the baseline's flattened Boolean `Gather` lookup.

## Result

- Baseline: 266/266, cost 949.
- Candidate: 266/266, cost 5066.
- Decision: reject for replacement. The candidate reduces parameters from 49 to 26, but explicit rotated tensors and concatenation increase memory from 900 to 5040. The baseline's one Gather is the better compiled representation of this fixed-size geometric rule.
