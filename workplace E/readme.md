# Workplace E

- Track: `public_source_ab_testing`
- Assignment: 66 primary tasks plus shared review of `task233`
- Source of truth: [`../assignments/task_assignment_400.csv`](../assignments/task_assignment_400.csv)
- Competition status: archived

## Main records

- [`TEAM_HANDOFF_20260715.md`](./TEAM_HANDOFF_20260715.md): accepted/rejected task chain, refs, hashes and commands as of the handoff.
- [`worklog.md`](./worklog.md): chronological experiments and Kaggle results.
- [`e_batch3_hidden_audit_20260715.json`](./e_batch3_hidden_audit_20260715.json): hidden-regression isolation.
- [`e_team_component_scan_20260715.json`](./e_team_component_scan_20260715.json): teammate package component scan.
- [`../docs/postmortem/2026-neurogolf-retrospective.md`](../docs/postmortem/2026-neurogolf-retrospective.md): final team result and cross-team retrospective.

The dated handoff is a historical snapshot and therefore still reports the then-current `7387.15` team package. The final team best was later updated to ref `54736568`, score `7420.93`; see the postmortem evidence rather than rewriting the old chronology.

## Historical loop

E evolved from ROI selection, to fixed task order, to a stricter online loop:

```text
exact parent -> official examples -> differential fuzz
-> at most three changed tasks -> Kaggle COMPLETE
-> accept, isolate, or reject -> record hidden risk
```

Confirmed failures such as `task110` and `task050` remain documented and must not be treated as safe merely because released examples pass.

## Source rules

- Do not copy closed-source Kaggle notebook code.
- Record URL, author, license/terms evidence, retrieval date and exact hashes for public sources.
- Keep credentials, external datasets, generated ONNX and submission ZIPs out of Git by default.
- Document every deletion with full path and reason.
