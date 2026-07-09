# Workplace B

Owner scope: B task only.

Tasks:

task001, task008, task018, task019, task023, task024, task056, task057, task063, task068, task076, task083, task088, task090, task097, task101, task104, task123, task125, task128, task131, task134, task140, task143, task151, task161, task163, task170, task172, task175, task181, task185, task205, task208, task209, task212, task228, task242, task244, task245, task247, task255, task261, task266, task270, task277, task280, task285, task289, task291, task293, task295, task300, task308, task312, task313, task317, task318, task320, task328, task344, task350, task360, task368, task369, task377, task395

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

Changed B tasks:

`task008`, `task076`, `task090`, `task131`, `task161`, `task170`, `task208`, `task245`, `task247`, `task270`, `task277`, `task280`, `task328`, `task368`

Important files:

- `20260709_b_yusuketogashi_v176_positive_v1/submission.zip`: full Kaggle candidate.
- `20260709_b_yusuketogashi_v176_positive_v1/overrides/`: the 14 replacement ONNX files.
- `20260709_b_yusuketogashi_v176_positive_v1/reports/`: score comparison and assignment reports.
- `20260709_b_yusuketogashi_v176_positive_v1/README.md`: detailed method, gains, risks, and next B-only targets.

Next direction:

Focus on B high-cost tasks still not materially improved by blending: `task018`, `task285`, `task101`, `task350`, `task255`, `task023`, then `task209`, `task205`, `task377`, `task212`, `task228`, `task344`. Prefer hidden-safe rule-equivalent ONNX rewrites over train-example lookup tables.
