# B integration on team submission (1)

This folder integrates the two accepted B rewrites into the 400-task package
provided as `submission (1).zip`.

## Base audit

- Input SHA256: `FD3F94D7B7B38ACD0151F00A83AE041FA23F5A418CFD6AAA5E0FCF17D9173AB5`.
- File count: 400 root-level ONNX models.
- Local validation: 400/400 valid.
- Local score: `7276.472341`.
- Corresponding Kaggle score: `7276.61` (ref `54604279`).

## Overrides

- task205 v8: cost `4251 -> 2691`, gain `+0.457241`.
- task266 analytic v2: cost `311 -> 170`, gain `+0.603994`.
- Combined local gain: `+1.061236`.
- Integrated local score: `7277.533577`.

The final package changes only task205 and task266. All other 398 model hashes
match the input package. The submitted artifact is `submission/submission.zip`.

Kaggle submission ref: `54605141`, public score `7277.67` (`COMPLETE`). The
online gain over the input package is `+1.06`.
