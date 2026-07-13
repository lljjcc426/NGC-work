# C+D Task Strategy Index

This index covers the 67 C tasks and the 67 mirrored D tasks.

## Task Sources

- C ownership: `assignments/task_assignment_400.csv`
- C task manifest: `workplace C/task_manifest_C.md`
- D mirror manifest: `workplace C/tasks_D/task_manifest_D.csv`
- D task JSON mirror: `workplace C/tasks_D/`

## Current Score Work

- C local campaign: `38_LOCAL_TASK_PLAN_PROGRESS_20260712.md`
- 7278.75 rebase: `39_SUBMISSION5_REBASE_RETROSPECTIVE_20260712.md`
- C+D 134-task benchmark: `41_CD_NEUROGOLF7300_BENCHMARK.csv`
- One-round derived optimization: `42_CD_ARCHIVE_ONE_ROUND_RESULTS.csv`
- Structural method analysis: `43_CD_ARCHIVE_METHOD_INTEL_20260713.md`
- Compliance-held candidates: `44_CD_ARCHIVE_DERIVED_CANDIDATES.md`

## Execution Policy

1. Validate every task on all public train/test/arc-gen examples.
2. Compare against the current parent cost, not an outdated manifest cost.
3. Do not promote equal-cost or invalid candidates.
4. Do not package archive-derived candidates while `compliance_hold=true`.
5. Keep independent task-specific improvements, such as task075, in the normal
   experiment ledger.
