# workplaceA

This directory is restricted to the screenshot WorkA task assignment.

Task count: 67

Future optimization and replacement scans for WorkA should use only the tasks listed in `workA_tasks.txt`.

The WorkA model set is selected per task from the previous and current local
baselines. A model is selected only after full `train + test + arc-gen`
validation, using the higher local points score. See
`baseline_comparison_20260710.json` for the latest comparison.
