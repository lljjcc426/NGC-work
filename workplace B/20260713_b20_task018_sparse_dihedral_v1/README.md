# B task018 sparse dihedral rewrite

This folder contains a self-written replacement for B task018. It does not
blend or copy a public task018 model.

The reconstructed generator rule is:

- locate one or two complete templates made from a frequent fill color and
  three marker colors;
- separate template marker positions from remote target marker positions;
- compare both templates under all eight dihedral transforms;
- stamp the matching template at each target and remove the source templates.

The first implementation used dynamic `NonZero` coordinates and passed all
266 official examples, but `NonZero` is prohibited by the competition scorer.
The retained model replaces it with whitelist-safe fixed `TopK` extraction.
It also avoids uint8 `TopK`, which is known to fail online for this task.

Results against the accepted 7296.04 package:

- cost: `24360 -> 19047`;
- points: `14.8993022782 -> 15.1453351122`;
- task gain: `+0.2460328340`;
- validation: all 3 train, 1 test, and 262 arc-gen examples;
- projected local package total: `7296.1494520737`.

The gain is retained for accumulation and was not submitted because it is
below the team's `+1.0` direct-submit threshold.

Contents:

- `models/task018.onnx`: validated replacement model;
- `src/rewrite_task018_sparse_dihedral.py`: reproducible model builder;
- `submission/submission.zip`: 400-task continuation package.

