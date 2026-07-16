# Contributing

## Before work

1. Pull the current `main` and record its commit.
2. Read `assignments/task_assignment_400.csv` for ownership.
3. Identify the exact champion parent by Kaggle ref and ZIP/model SHA.
4. Read the task's accepted record, semantic rule, failed experiments, and hidden-risk notes.

## Experiment boundary

New work uses one directory per task as described in
[`docs/repository-guide.md`](docs/repository-guide.md). A worker may change only its task unit and disposable scratch files. Shared score tables, source manifests, and trick registries are updated after validation, not concurrently by every worker.

## Promotion gate

A candidate is accepted only when all applicable checks pass:

1. Reproducible builder succeeds from a clean output directory.
2. ONNX checker, strict shape inference, required names/shapes and runtime load pass.
3. Every official released example passes exactly.
4. Fresh generator or documented differential fuzz passes.
5. Official cost is lower on the exact current parent.
6. Model and package file limits, names, counts, CRC and SHA are recorded.
7. Risky changes receive isolated or attributable Kaggle confirmation.

An online regression, timeout, 0 score, higher cost, or unsupported operator is recorded as `rejected`; do not silently overwrite the failed evidence.

## Commit scope

- Commit source, compact results, manifests and conclusions.
- Do not commit generated ONNX, submission ZIPs, downloaded datasets, traces or caches by default.
- Never commit credentials, cookies, tokens, `.env`, `kaggle.json` or access-token files.
- Keep unrelated member work intact; do not revert or reorganize another workplace without coordination.
- List every deleted file and reason in the commit message or worklog.

## Naming

- Task: `taskNNN`, zero-padded to three digits.
- Dated record: `YYYYMMDD_<team>_<purpose>_vN`.
- Kaggle description: parent ref, changed tasks, expected delta, validation level and SHA prefix.
- Status: `candidate`, `local_valid`, `fuzz_valid`, `online_confirmed`, `rejected`, or `research_only`.

## Pull request checklist

- [ ] Parent ref and SHA recorded.
- [ ] Changed task list is explicit.
- [ ] Builder and result record agree with the model hash.
- [ ] Local and generated validation counts recorded.
- [ ] Expected and observed Kaggle deltas recorded when submitted.
- [ ] Public source metadata and license/terms evidence recorded.
- [ ] No credentials or unintended binaries are tracked.
- [ ] `python tools/repository_audit.py` passes.
- [ ] Deletions, if any, are listed explicitly.
