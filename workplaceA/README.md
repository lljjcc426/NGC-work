# workplaceA

This directory is restricted to the screenshot WorkA task assignment.

Task count: 67

Future optimization and replacement scans for WorkA should use only the tasks listed in `workA_tasks.txt`.

The WorkA model set is selected per task from the previous and current local
baselines. A model is selected only after full `train + test + arc-gen`
validation, using the higher local points score. See
`baseline_comparison_20260710.json` for the latest comparison.

The selected WorkA set is now synchronized into the current local
`submission.zip`; all 67 WorkA task models match the package.

Latest verified optimization: `task148.onnx` reuses `zero_rows_b` for the
all-False middle block, reducing params from 252 to 108 with +0.0299087714
local points. See `worklog_task148_20260711.md`.
