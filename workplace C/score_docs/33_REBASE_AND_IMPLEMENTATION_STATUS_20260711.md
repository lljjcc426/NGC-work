# C Rebase And Implementation Status 2026-07-11

> Superseded by `34_FULL_RETROSPECTIVE_AND_NEXT_BUILD_PLAN_20260711.md`.
> The v90 package was later supplied by the user, task158 was rebased, and the
> next visible score became 7271.95.

## Current Team Best

- competition: `neurogolf-2026`
- team: `Blacklions.`
- team_id: `16252365`
- Kaggle submission id: `54557194`
- public score: `7271.93`
- description: `v90 compact Pad axes 21 tasks local +0.033332 sha 02f7787e`
- recovery status: `BLOCKED`

Kaggle CLI commands used:

```powershell
kaggle competitions submissions neurogolf-2026 --page-size 100 --format json
kaggle competitions leaderboard neurogolf-2026 --show --download -p data\manifests\leaderboard_download
kaggle competitions team-submissions 16252365 --format json
kaggle kernels output muelsyse111/neurogolf-submit-current -p ...
kaggle kernels output muelsyse111/neurogolf-7113-franksunp-payload-submit -p ...
kaggle kernels output muelsyse111/neurogolf-submit-prvsiyan-7266-72-repro -p ...
```

Result: CLI confirms the team best submission but does not expose a command to download the submitted `submission.zip` by submission id. The local zip scan found no `submission.zip` with SHA prefix `02f7787e`.

Full recovery evidence: `workplace C/score_docs/32_TEAM_BEST_REBASE_RECOVERY.md`.

## Implemented This Round

### task372 common-mode grouped Conv

- script: `workplace C/single_task/task372/scripts/build_common_mode_grouped.py`
- debug ONNX: `workplace C/single_task/task372/debug/task372_common_mode_probe.onnx`
- result: `REJECTED`

The model cost drops from `710` to `80`, but official validation fails `0/266`.

Root cause: the proof was argmax-equivalent only. Official `run_network` uses per-channel `output > 0.0`, so subtracting a class-common logit changes the one-hot threshold output.

### task356 zero-tie fold

- script: `workplace C/single_task/task356/scripts/build_zero_tie_fold.py`
- debug ONNX: `workplace C/single_task/task356/debug/task356_zero_tie_probe.onnx`
- result: `REJECTED`

The model cost drops from `1319` to `1218`, but official validation fails `0/266`.

Root cause: deleting `Add(mask10, mask10)` makes the background class tie at zero. Official thresholding then outputs no positive class, not color 0.

### task349 old candidate recheck

- candidate: `workplace C/single_task/task349/onnx/task349_candidate.onnx`
- result: `REJECTED`
- validation: `6/267`
- cost: `10362`

This is not a usable candidate despite lower cost.

### task158 existing positive candidate

- candidate: `workplace C/single_task/task158/onnx/task158_candidate.onnx`
- validation: `266/266`
- old cost on prvsiyan/v86 parent: `28483`
- new cost: `28023`
- delta points: `+0.01628181651604521`
- status: `usable only after highest-parent rebase`

`task158` is identical between prvsiyan 7266 and v86 (`b1917c261...`), so the candidate is a real local improvement on those parents. It has not been proven against v90 because v90 package recovery is blocked.

## Submit Gate

No Kaggle submission was made.

Reason: user requested rebase onto the current team best before submitting. Current team best `54557194` / `7271.93` / `sha 02f7787e` could not be recovered from Kaggle CLI or local zip cache. Submitting from v86 would be below the current team best and would waste quota.

## Next Execution Target

1. Recover v90 package by obtaining the actual `submission.zip` for submission `54557194` or by finding the missing local zip with SHA prefix `02f7787e`.
2. Once recovered, check `task158.onnx` SHA and cost in v90.
3. If v90 `task158` is still `b1917c261...`, rebase `task158_candidate.onnx` onto v90 and submit one positive probe.
4. If v90 `task158` differs, score v90 task158 first and only apply the candidate if it lowers cost.
5. Continue new `task349` codebook search separately; do not reuse the invalid old candidate.
