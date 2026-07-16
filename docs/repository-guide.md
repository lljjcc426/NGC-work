# Repository guide

## Goal

This repository is both a competition archive and a reusable example of task-level ONNX optimization. The structure keeps historical paths stable while giving future work a single entry point and a reproducible record format.

## Stable areas

| Path | Ownership | Policy |
| --- | --- | --- |
| `assignments/` | team | Assignment snapshots; CSV is authoritative. |
| `docs/` | team | Cross-team conclusions, evidence, templates, and postmortems. |
| `neurogolf_400_tasks/` | team | Immutable public task mirror. |
| `tools/` | team | Repository maintenance only; no private data dependency. |
| `workplace A` ... `workplace F` | member | Historical experiment records and member-owned scripts. |
| `.github/workflows/` | team | Reproducibility and safety checks. |

Existing workplace paths are frozen. Do not bulk-move or rename historical files merely for style; old Kaggle descriptions, commits, and Markdown links refer to them.

## Canonical data and known archive debt

- `neurogolf_400_tasks/tasks/taskNNN.json` is the canonical public task mirror for new tools.
- `workplace C/tasks/` and `workplace C/tasks_D/` contain historical task copies. They are retained because old manifests and scripts reference them; do not create further copies.
- The repository already tracks historical ONNX and ZIP files that match current ignore rules. They remain as competition evidence, while new generated binaries are ignored by default.
- Historical scripts contain local Windows paths. New code resolves from the repository root or documented environment variables and must not add machine-specific defaults.
- Dated experiment directories are immutable snapshots. Corrections belong in a later record that links back to the original.

## New task unit

New work belongs under the responsible workplace:

```text
workplace X/
  tasks/
    taskNNN/
      README.md       semantic rule, assumptions, history, hidden risks
      builder.py      reproducible ONNX builder and task-local checks
      result.json     machine-readable accepted state and evidence
      candidates/     optional source-only experiments
```

Generated `taskNNN.onnx`, submission ZIPs, runtime traces, downloaded datasets, and caches remain outside Git. If a binary must be retained for exact reproduction, record its SHA256, origin, license, and reason before force-adding it.

## Evidence levels

Use one of these labels in `result.json` and task notes:

| Label | Meaning |
| --- | --- |
| `candidate` | Builds, but has not passed all local gates. |
| `local_valid` | Passes official examples and scorer on the exact parent. |
| `fuzz_valid` | Also passes the documented generated/differential tests. |
| `online_confirmed` | Isolated or attributable Kaggle score agrees with the expected delta. |
| `rejected` | Incorrect, higher cost, runtime-fragile, or online regression. |
| `research_only` | Preserved for analysis and explicitly forbidden from packaging. |

An accepted model always names its parent ref/SHA, builder, model SHA, cost components, validation counts, expected delta, and Kaggle ref where available.

## Source policy

- Public source use requires the URL, author, license/terms evidence, retrieval date, and exact files or hashes used.
- Closed-source notebook code is not copied into the repository.
- Private Kaggle datasets/notebooks remain private and are referenced only by metadata needed for reproduction.
- Credentials, cookies, tokens, `.env`, `kaggle.json`, and access-token files are never committed.
- A public candidate is not accepted merely because it scores lower locally; it must pass the same correctness and hidden-risk gates as an original rewrite.

## Archive policy

- Negative results stay in logs because they prevent repeated hidden-set failures.
- Dated handoffs remain historical snapshots. Later corrections link to them instead of rewriting their original chronology.
- File deletion requires path, reason, size when relevant, and commit/worklog disclosure.
- Temporary probes may be deleted only after their conclusion is recorded.

## Automated checks

Run:

```powershell
python tools/repository_audit.py
```

The audit verifies required entry points, the 400-primary/402-slot assignment invariant, 67 slots per owner, UTF-8 text, local Markdown links, and tracked credential filenames. Existing historical binaries are reported as warnings rather than retroactively rejected.
