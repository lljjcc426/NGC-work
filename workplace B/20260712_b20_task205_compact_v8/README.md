# B task205 compact rewrite v8

This folder contains a B-only, independently written rewrite of the current
online-safe `task205.onnx`. The final model is `model/task205.onnx`.

## Result

- Baseline cost: `4251` (`16.645090` points).
- Final cost: `2691` (`17.102332` points).
- Single-task gain: `+0.457241`.
- Cost reduction: `1560` (`36.70%`).
- Validation: all 266 official examples plus 8192 seeded random legal grids.
- Raw output equivalence to the online-safe baseline: `8458/8458` exact.

## Main rewrites

1. Move float16 casts before Gather and let broadcasted Mul replace redundant
   Squeeze/Unsqueeze chains.
2. Remove fixed batch dimensions from the color and coordinate Einsums so two
   coordinate stacks can be represented as `[2, 30]`.
3. Replace `1 - Cast(bool)` with `Not(bool) -> Cast(float16)`.
4. Keep coordinate addition, clipping, span comparison, and Gather indices in
   `int32`, removing two full `[1, 30]` casts.
5. Replace background exclusion by multiplication with one `Where` operation.
6. Keep row and column coordinate construction boolean through Gather, And,
   and Concat, then cast each completed matrix to float16 only once.

The scripts are ordered v1 through v8 and reproduce the final model from the
online-safe v6 task205 baseline. `reports/score_report.json` contains the full
cost progression; `reports/equivalence.json` contains the strict regression
result.

Together with the accepted task266 analytic rewrite, this result exceeds the
team rule of `+1.0`; the two overrides were submitted as one B-only aggregate.

## Online probe

- Package: `submission/submission.zip`.
- Changed tasks: task205 and task266 only; the other 398 hashes match the base.
- Combined local gain: `+1.061236`.
- Kaggle submission ref: `54604712`.
- Initial status: `PENDING`.
