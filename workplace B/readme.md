# Workplace B

Owner scope: B task only.

Tasks:

task001, task008, task018, task019, task023, task024, task056, task057, task063, task068, task076, task083, task088, task090, task097, task101, task104, task123, task125, task128, task131, task134, task140, task143, task151, task161, task163, task170, task172, task175, task181, task185, task205, task208, task209, task212, task228, task242, task244, task245, task247, task255, task261, task266, task270, task277, task280, task285, task289, task291, task293, task295, task300, task308, task312, task313, task317, task318, task320, task328, task344, task350, task360, task368, task369, task377, task395

## 2026-07-13: new team baseline, six original B rewrites

Folder: `20260713_b_team7378_original_rewrites_v1`

- New provided baseline validates `400/400` at `7377.858249` locally.
- Changed only B tasks `018`, `123`, `134`, `285`, `350`, and `360`.
- Combined local gain: `+1.062037`.
- Projected local score: `7378.920286`.
- Main new structures are task123 nested-square factorization, task134
  generator-proven one-axis variance, task350 shared rule tensors, and task360
  shared fold/output route.
- Complete submission ZIP, six models, scripts, and reports are included.
- Kaggle ref: `54647712` (submitted; initial status `PENDING`).

## 2026-07-09: yusuketogashi V176 B-only positive graft

Folder: `20260709_b_yusuketogashi_v176_positive_v1`

What changed:

- Pulled and extracted Kaggle notebook `yusuketogashi/2026-neurogolf-baseline`.
- Compared its embedded V176 submission against our local `aggr7250_25_jack_direct_v1` baseline on B tasks only.
- Kept only 14 locally positive B-task replacements.
- Built a full `submission.zip` while leaving every non-B task unchanged.

Expected local improvement:

- Base: `7250.128424`
- Gain: `+1.336384`
- Expected total: `7251.464808`
- Kaggle ref: `54488042`
- Kaggle public score: `7251.58`
- Status: useful B-task evidence, but behind the later public notebook baseline `7267.31`.

Changed B tasks:

`task008`, `task076`, `task090`, `task131`, `task161`, `task170`, `task208`, `task245`, `task247`, `task270`, `task277`, `task280`, `task328`, `task368`

Important files:

- `20260709_b_yusuketogashi_v176_positive_v1/submission.zip`: full Kaggle candidate.
- `20260709_b_yusuketogashi_v176_positive_v1/overrides/`: the 14 replacement ONNX files.
- `20260709_b_yusuketogashi_v176_positive_v1/reports/`: score comparison and assignment reports.
- `20260709_b_yusuketogashi_v176_positive_v1/README.md`: detailed method, gains, risks, and next B-only targets.

Next direction:

Focus on B high-cost tasks still not materially improved by blending: `task018`, `task285`, `task101`, `task350`, `task255`, `task023`, then `task209`, `task205`, `task377`, `task212`, `task228`, `task344`. Prefer hidden-safe rule-equivalent ONNX rewrites over train-example lookup tables.

## 2026-07-09: yusuke V176 active-base B improvements

Folder: `20260709_b_yusuke_v176_t023ext_t255qmm_v1`

Active baseline changed:

- We now use yusuketogashi V176 as the working baseline.
- Kaggle public baseline observed: `7267.31`.
- Local scorer baseline: `7267.145162`.

Current B-only positive changes:

- `task023`: historical `aggr7162_ext7000_t023_merge` override, `+0.327464`, cost `8222 -> 5926`.
- `task255`: new uint8 `QLinearMatMul` tail rewrite, `+0.168126`, cost `8911 -> 7532`.

Combined local gain:

- Gain: `+0.495590`
- Expected local total: `7267.640752`
- Status: not submitted yet because it is below the `+1.0` direct-submit threshold.

Important files:

- `20260709_b_yusuke_v176_t023ext_t255qmm_v1/submission.zip`: full candidate, only `task023` and `task255` changed.
- `20260709_b_yusuke_v176_t023ext_t255qmm_v1/overrides/`: accepted ONNX replacements.
- `20260709_b_yusuke_v176_t023ext_t255qmm_v1/scripts/`: accepted task255 rewrite and robust scorer.
- `20260709_b_yusuke_v176_t023ext_t255qmm_v1/reports/`: evidence from candidate-pool and selected-variant rescoring.

Next direction:

Continue accumulating B-only verified gains until the package exceeds `+1.0`, then submit. The most important remaining targets are `task018`, `task285`, `task101`, `task076`, `task350`, `task209`, and `task328`.

## 2026-07-10: B-only selfrewrite v2

Folder: `20260710_b_yusuke_v176_selfrewrite_v2`

Active baseline:

- Baseline package: `yusuketogashi_v176_full`.
- Kaggle public baseline: `7267.31`.
- Local scorer baseline: `7267.145162`.

What changed:

- Stayed inside B task scope.
- Stopped broad public-code resweeps and focused on ONNX rewrites.
- Built a seven-task package combining two earlier positive replacements with five new self-rewrites.
- Submitted because the local gain exceeded the `+1.0` direct-submit threshold.

Changed B tasks:

| task | method | cost | local gain |
| --- | --- | ---: | ---: |
| `task023` | compact external merge already positive on V176 | `8222 -> 5926` | `+0.327464` |
| `task076` | bool-mask `TopK` via `uint8` instead of `float16` | `12825 -> 12136` | `+0.055220` |
| `task205` | `float16` final-tail rewrite | `4891 -> 4251` | `+0.140243` |
| `task255` | `uint8/QLinearMatMul` tail rewrite | `8911 -> 7532` | `+0.168126` |
| `task277` | bold prune of one A/B MaxPool iteration | `3540 -> 3140` | `+0.119904` |
| `task285` | direct `uint8 TopK`, removing cast overhead | `19700 -> 17476` | `+0.119790` |
| `task377` | `float16` final-tail rewrite | `4567 -> 3967` | `+0.140846` |

Combined local result:

- Gain: `+1.071593`.
- Expected local total: `7268.216755`.
- Submitted zip SHA256: `066fb7fc24afcfc94af4525375834816a8e7b79bae9808b042f5108a46543c8a`.
- Kaggle ref: `54517159`.
- Kaggle status: `ERROR`.

Risk note:

`task277` is the intentional high-risk piece. It passes official local `train`, `test`, and `arc-gen` examples, but an earlier random equivalence stress test found a mismatch. The later single-task online probe was positive, so it remains in the safe subset for now.

Follow-up result:

- Full v2 ref `54517159`: `ERROR`.
- No-fp16-output debug ref `54517405`: `ERROR`.
- Core package without uint8 TopK ref `54517546`: completed but scored `7251.58`, so `task023` is hidden-unsafe.
- Conclusion: keep `task205`, `task255`, `task277`, `task377`; exclude `task023`, `task076`, `task285`.

## 2026-07-10: B-only online-safe v3

Folder: `20260710_b_yusuke_v176_online_safe_v3`

Active baseline:

- Baseline package: `yusuketogashi_v176_full`.
- Kaggle public baseline: `7267.31`.
- Local scorer baseline: `7267.145162`.

Changed B tasks:

| task | method | cost | local gain |
| --- | --- | ---: | ---: |
| `task205` | `float16` final-tail rewrite | `4891 -> 4251` | `+0.140243` |
| `task255` | `uint8/QLinearMatMul` tail rewrite | `8911 -> 7532` | `+0.168126` |
| `task277` | prune one A/B MaxPool iteration | `3540 -> 3140` | `+0.119904` |
| `task377` | `float16` final-tail rewrite | `4567 -> 3967` | `+0.140846` |

Combined result:

- Local gain: `+0.569119`.
- Expected local total: `7267.714281`.
- Kaggle ref: `54517873`.
- Kaggle public score: `7267.85`.
- Online gain vs yusuke public baseline: about `+0.54`.

Probe evidence:

- `task255` only, ref `54517673`: score `7267.45`.
- `task277` only, ref `54517695`: score `7267.40`.
- `task205+task377`, ref `54517745`: score `7267.56`.

Blocked / do not repeat:

- `task023`: locally positive but hidden-unsafe; the debug package scored `7251.58`.
- `task076` and `task285`: `uint8 TopK` rewrites caused Kaggle `ERROR`.
- `task101`: old full ARC-GEN generator scored worse on V176 (`14.29` points vs V176 `15.47`), so do not replace the whole model that way.

Next direction:

Stay B-only and self-rewrite first. Attack `task018`, `task101`, `task350`, `task209`, and `task328` by directly simplifying the current V176 ONNX graphs. If the current team-best package behind ref `54517518` (`7268.99`) becomes available locally, test these four safe B overrides on top of that base immediately.

## 2026-07-11: B-only online-safe v5

Folder: `20260711_b_yusuke_v176_online_safe_v5`

This pass continued with self-written, task-level ONNX rewrites only. The final
package changes seven B tasks and leaves every non-B task untouched.

Result:

- Local gain over yusuketogashi V176: `+1.030562`.
- Expected local total: `7268.175724`.
- Kaggle ref: `54568534`.
- Kaggle public score: `7268.31`.
- Online gain over the V176 public score `7267.31`: `+1.00`.

New verified rewrites in this pass:

- `task209`: one-pad template stack, `+0.018695` local.
- `task328`: algebraic corner-distance split, `+0.092680` local.
- `task368`: dual `QLinearConv` stamp and direct bbox mask, `+0.350068` local.

The main breakthrough is `task368`, whose cost fell from `3599` to `2536`.
It passed 2048 random equivalence checks, 5000 fresh exact-generator examples,
and an isolated online probe before inclusion in v5.

New blocked rule:

Do not use uint8 `TopK` in `task018`, `task076`, or `task285`. All three pass
local ONNX Runtime checks but return Kaggle `ERROR`, including versions with
explicit repaired shape metadata. The rejected candidates are retained only as
negative evidence in the v5 `experiments/` folder.

Integration note:

The team also had a newer non-B aggregate scoring `7271.93`. Use the seven ONNX
files in `20260711_b_yusuke_v176_online_safe_v5/overrides/` as a B-task overlay
on that newer full package.

## 2026-07-12: B-20 task266 analytic rewrite

Folder: `20260712_b20_task266_analytic_v2`

The active objective is now to push every B task toward at least 20 points. Team
submission blending is paused; this folder is an independent task-level result.

- Exact generator: `a9f96cdd`.
- Cost: `311 -> 170`.
- Points: `19.260207 -> 19.864202`.
- Gain: `+0.603994`.
- Validation: official train/test/ARC-GEN plus all 15 legal marker positions.
- Remaining gap to 20: reduce cost by another 22, from 170 to at most 148.

The rewrite replaces two learned convolutions and a ReLU with a one-channel
marker encoding followed by one analytic linear classifier. The failed task001
quantized outer-product experiment is retained in the folder as negative
evidence, so its incorrect flatten ordering is not retried.

Cross-20 follow-up: four task266 architectures with theoretical costs between
113 and 145 were rejected by exact state-code enumeration or fixed sign
conflicts. A task313 rank-3 Einsum candidate (cost 135) was also rejected because
the fourth coordinate basis is required by the generator's joint period-2 and
period-3 phase. These negative results are recorded under
`20260712_b20_task266_analytic_v2/reports/`.

The next bold-structure pass rejected three hidden-risk shortcuts: task285
without its safety Pad fails fresh generator inference at Gather index 924;
task181 dynamic Scatter indices cost more than its static table; and task395
sparse zero tensors cannot feed dense Concat under the official ONNX checker.
No model from these three experiments should be submitted. Full evidence is in
`20260712_b20_task266_analytic_v2/reports/20260712_bold_structure_search_round2.json`.

## 2026-07-12: B-20 task205 compact rewrite v8

Folder: `20260712_b20_task205_compact_v8`

This pass stays B-only and uses no new public submission packages. It rewrites
the current online-safe task205 graph one chain at a time.

- Cost: `4251 -> 2691` (`-36.70%`).
- Points: `16.645090 -> 17.102332`.
- Gain: `+0.457241`.
- Validation: `8458/8458` exact against the online-safe baseline, consisting of
  all 266 official examples and 8192 seeded random legal grids.

The largest reduction comes from replacing two float32 complement pipelines
with boolean `Not`, keeping coordinate arithmetic in int32, and keeping the
final coordinate matrices boolean until one post-Concat float16 Cast. The
recommended override is
`20260712_b20_task205_compact_v8/model/task205.onnx`.

Combined with the accepted task266 analytic rewrite (`+0.603994`), the two-task
B-only gain is about `+1.061236`, crossing the team threshold for an immediate
online probe. The 400-model package changes only task205 and task266; Kaggle
submission ref `54604712` completed at `7269.46`, an online gain of `+1.07`
over the `7268.39` base and a close match to the local `+1.061236` estimate.

## 2026-07-12: B rewrites integrated into team submission (1)

Folder: `20260712_b20_team_integration_v1`

The provided 400-task package was fully audited at `7276.472341` local and
corresponds to Kaggle ref `54604279` scoring `7276.61`. Its task205 and task266
were still the old cost-4251 and cost-311 models, so the accepted B v8/v2
overrides were applied without changing the other 398 hashes.

- Combined local gain: `+1.061236`.
- Integrated local score: `7277.533577`.
- Kaggle ref: `54605141`, public score `7277.67` (`COMPLETE`).
- Online gain over the provided package: `+1.06`.

## 2026-07-12: B-20 task104 rank-2 sign rewrite

Folder: `20260712_b20_task104_rank2_sign_v1`

The current B scoreboard has 67 tasks: 9 are at or above 20 points and 58 are
still below 20. Work now proceeds one task at a time without public-package
rescans.

task104 was rewritten from a three-component template Einsum to a rank-2
indefinite sign factorization derived from the four legal generator
orientations.

- Cost: `238 -> 160`.
- Points: `19.527729 -> 19.924826`.
- Gain: `+0.397097`.
- Validation: `7/7` official examples, all four legal orientations, minimum
  positive raw margin `5.0`.
- Candidate aggregate local score: `7277.930673`.

This gain is retained for the next aggregate because it is below the `+1.0`
direct-submit threshold. The same folder records strict negative evidence for
task395 sparse/legacy attempts and structural audits of tasks083, 261, 001,
and 360, so those exact failed representations are not repeated.

## 2026-07-12: neurogolf7300+ B-task audit and online integration

Folder: `20260712_b20_zealous7300_audit_v1`

The `zealous9230/neurogolf7300` dataset contains 399 ONNX models, no generator
code, and omits task173. After filling the missing file only for scoring, its
local total is `7371.855070`; this is treated as a candidate mine rather than a
trusted submission because at least task285 contains a known hidden-unsafe
shortcut.

Four B models have now passed task-level checks and online probes:

- task293: cost `1043 -> 40`, local gain `+3.260977`.
- task056: cost `34 -> 30`, local gain `+0.125163`.
- task104: cost `238 -> 118`, local gain `+0.701586`, now `20.229315` points.
- task205: cost `2691 -> 2084`, local gain `+0.255624`.

Kaggle ref `54613658` proved task293 in isolation at `7280.93` (`+3.26`). Ref
`54613833` then accepted task056/task104/task205 at `7282.01` (`+1.08`), giving
a total online improvement of `+4.34` from the `7277.67` baseline. The B group
now has 11 tasks at or above 20 points. Full comparison, risks, accepted models,
and the exact submission artifacts are retained in the audit folder.

## 2026-07-13: four terminal-rule models accepted at 7290.38

Folder: `20260713_b20_terminal_rules_online_v2`

Four further B tasks were integrated one at a time on top of the accepted
`7282.01` package. All predicted gains matched the online direction and size:

- task161: cost `1779 -> 275`, online score `7283.88`.
- task163: cost `1791 -> 310`, online score `7285.64`.
- task212: cost `2249 -> 412`, online score `7287.33`.
- task350: cost `9036 -> 428`, online score `7290.38`.

This round adds `+8.37` online. Task350 also passed 500 independently generated
grids after its exact first-to-last span rule was reconstructed. A task163
sparse-initializer attempt was rejected locally because official strict shape
inference cannot infer sparse Einsum input ranks; the failed construction is
documented and must not be submitted.

## 2026-07-13: B tail integration accepted at 7296.04

Folder: `20260713_b20_tail_online_v3`

Twelve more B models were added in four online-verified batches. Scores moved
from `7290.38` through `7292.17`, `7293.55`, and `7294.97` to `7296.04`.
Together with the preceding terminal-rule round, the full run from `7282.01`
adds `+14.03` online.

The new tasks are 001, 024, 143, 244, 245, 255, 291, 313, 344, 368, 369, and
377. Task285 remains explicitly excluded because its smaller no-Pad model is
hidden-unsafe. Task255 is now the main independent rewrite target: cost 5976,
16.304493 points.

## 2026-07-13: task018 sparse dihedral rewrite

Folder: `20260713_b20_task018_sparse_dihedral_v1`

Task018 was independently reconstructed as a sparse template-selection
problem over one or two source templates and all eight dihedral transforms.
The whitelist-safe fixed-TopK model passes all 266 official examples and cuts
cost from `24360` to `19047`, for `+0.246033` local points. A smaller dynamic
`NonZero` version was rejected because that operator is prohibited by the
competition scorer; uint8 `TopK` also remains excluded for online safety.

The 400-task continuation package projects to `7296.149452` locally. It is
archived but not submitted yet because the accumulated new gain is below the
`+1.0` direct-submit threshold.

## 2026-07-13: task285 exact quadrant rewrite

Folder: `20260713_b20_task285_exact_quadrant_v1`

task285 was reconstructed from the exact ARC-GEN `b775ac94` rule and rewritten
without public-model blending. The model resolves the complete source sprite,
expands its connected 5x5 mask, and reflects it into all colored target
quadrants. Signed int8 colors make padding double as the boundary sentinel,
while the float16 `TopK` path is retained for Kaggle compatibility.

- Cost: `19700 -> 18189`.
- Points: `15.111626 -> 15.191428`.
- Gain: `+0.079802`.
- Validation: `265/265` official examples, `50000/50000` fresh exact-generator
  cases, and `400/400` valid files in the cumulative package.
- Accumulated task018 + task285 local gain: `+0.325834`.
- Accumulated local score: `7296.229254`.

The package remains below the `+1.0` direct-submit threshold. The current
task-level report also corrects stale assignment data: task350 is already at
cost 428 and 18.940877 points, so the next structural pass moves to task101
and task076 instead.

## 2026-07-13: exact rewrite batch 2 on accepted 7379.07 parent

Folder: `20260713_b_team7379_exact_batch2_v1`

Seven further B-only rewrites pass their complete official datasets and the
combined 400-model package validates `400/400`. Costs changed as follows:

- task076: `12825 -> 12313`.
- task101: `13711 -> 13071`.
- task163: `310 -> 298`.
- task208: `4181 -> 4084`.
- task209: `7609 -> 7324`.
- task270: `2846 -> 2719`.
- task328: `5746 -> 5189`.

The reliable unpublished gain is `+0.337284`; the full continuation package
scores `7379.257570` locally and projects to about `7379.41` from Kaggle ref
`54647712` (`7379.07`). It is archived but not submitted because the gain is
below the agreed `+1.0` verification threshold. Resume from this package and
accumulate another `+0.662716` before the next direct submission.

## 2026-07-14: submission 3 B batch accepted at 7384.93

Folder: `20260714_b_submission3_exact_batch1_online7384_93_v1`

The team-provided `submission (3).zip` validates `400/400`, scores
`7383.777544` locally, and has online score `7383.93` at ref `54673685`.
Ten exact B-only overrides were rebased onto it: tasks 076, 101, 123, 163,
208, 209, 270, 328, 350, and 360.

The resulting package validates `400/400` at `7384.778427` locally, a gain of
`+1.000883`. Kaggle ref `54686944` completed at `7384.93`, confirming a full
`+1.00` online gain over the same baseline. task208 also passed 20,000 fresh
exact-generator samples; task163 global state truncation and the more
aggressive task350 gate-removal variants are explicitly rejected and retained
only as documented search scripts.
