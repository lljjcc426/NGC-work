# B-only selfrewrite v4 diagnostic bundle

This was the first `+1` local bundle from the 2026-07-11 rewrite pass.

- Local gain: `+1.025054`.
- Expected local total: `7268.170216`.
- Kaggle ref: `54568105`.
- Kaggle status: `ERROR`.

The failure was isolated to `task018`'s uint8 `TopK`. The other new work was
retained and rebuilt as `../20260711_b_yusuke_v176_online_safe_v5/`, which
completed at `7268.31`. Use v5, not this submission zip.

This folder remains as an audit trail for the original package and reports.

