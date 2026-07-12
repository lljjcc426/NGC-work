# C Deep Individual Modeling Campaign

Generated: 2026-07-12

## Result

- Strict deep-model coverage: `67/67` C tasks.
- Completion evidence: `C_DEEP_MODEL_STATUS.csv` and `C_DEEP_MODEL_STATUS.md`.
- Required per task: a rule report, a structurally different builder, a candidate ONNX, and an official cost-diff record.
- Metadata-only, opset-only, identity, padding-constant-only, and generic optimizer attempts do not qualify.
- New accepted local improvement in this campaign: `task298`, cost `135 -> 129`, full validation `267/267`.
- Online verification: submission `54595725`, score `7273.42`, parent score `7273.37`, observed delta `+0.05`.

## Rebase And Submission

- Parent archive: user-provided `E:/submission (3).zip`.
- Parent SHA256: `d3284267c02846dde8571890d4c761dcf9592fce2ec190c3348a0dee1c13c44f`.
- The SHA prefix exactly matches the team-best v93 submission description.
- Parent file count: `400` ONNX models.
- The parent task148 graph contained duplicate node names. The local gate repair renamed one duplicate node only; graph topology, operators, parameters, and outputs were unchanged.
- Candidate: `GOLF_20260712_099_v93_plus_task298`.
- Local submission validation: `1197/1197` smoke examples, `400/400` files.
- Kaggle result: `7273.42`, status `COMPLETE`.

## Final Ten Tasks

| task | independent model | validation | old cost | new cost | delta cost | decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| task383 | crop then color decode | 266/266 | 5830 | 33497 | +27667 | reject |
| task382 | shared color projection map | 266/266 | 5665 | 16495 | +10830 | reject |
| task165 | explicit float template correlation | 265/265 | 4532 | 6604 | +2072 | reject |
| task378 | broadcast coordinate moments | 267/267 | 3089 | 75149 | +72060 | reject |
| task132 | direct grid axis reductions | 267/267 | 3652 | 3899 | +247 | reject |
| task069 | explicit float dynamic correlations | 264/264 | 2948 | 4644 | +1696 | reject |
| task284 | shared marker projection map | 266/266 | 3085 | 39099 | +36014 | reject |
| task201 | shared frame projection map | 266/266 | 3045 | 39059 | +36014 | reject |
| task224 | shared gray projection map | 266/266 | 1886 | 37900 | +36014 | reject |
| task094 | slice/reduce line detector | 265/265 | 2677 | 5856 | +3179 | reject |

All ten candidates implement the task rule with a genuinely different computation graph and pass every public train/test/arc-gen example. They are retained as measured counterexamples and are not grafted because cost increased.

## Engineering Conclusions

1. Materializing a full `[channel, row, col]` selection map is consistently expensive. It added roughly `36k` cost in task201, task224, and task284.
2. Compact Einsum projections are already efficient for row/column occupancy and moments. Replacing them with Mul plus ReduceSum usually increases activation memory.
3. QLinearConv is materially cheaper than an exact Cast/float-Conv/Cast expansion for task069 and task165.
4. task132 is the closest rejected rewrite: direct axis reduction was exact and only `247` cost worse. A useful next attempt must eliminate other projection parameters at the same time.
5. task298 confirms that fusing selection into an existing Einsum can reduce both cost and public score as predicted; this is preferable to creating new intermediate tensors.

## Reproduction

```powershell
python "workplace C/neurogolf-2026-work/scripts/c_refresh_deep_model_status.py"
python "workplace C/single_task/task094/scripts/build_deep_model.py"
```

Each task has the same independent builder interface under `workplace C/single_task/taskXXX/scripts/`.
