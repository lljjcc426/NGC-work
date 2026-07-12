# task311 independent modeling

## Rule

The logical input is 3x3. Each output row is the original three cells followed by those same cells in reverse order, producing a 3x6 horizontal reflection: `output[r] = input[r] + reverse(input[r])`.

## Independent structure

`scripts/build_horizontal_reflection.py` crops the 3x3 logical patch, reverses its width using a negative-step Slice, concatenates original and reflection along width, and pads the 3x6 result to 30x30. This replaces the baseline's fixed 30-index Gather with explicit geometry.

## Official validation and cost

| variant | passed | checked | memory | params | cost |
|---|---:|---:|---:|---:|---:|
| baseline Gather | 266 | 266 | 0 | 30 | 30 |
| Slice/reverse/Concat | 266 | 266 | 1440 | 18 | 1458 |

The model is exact and parameter-lighter, but the reflected patch and concatenated output are counted as intermediates. It is retained as an independent solver, not selected for replacement.
