# B-20 task266 analytic rewrite

This is an independent B-task result. It is not blended with the team-best
submission package.

## Result

- Task: `task266` (`a9f96cdd`)
- Baseline cost: `311`
- New cost: `170`
- Baseline points: `19.260207`
- New points: `19.864202`
- Local gain: `+0.603994`
- Remaining cost reduction needed for 20 points: `22`

## Rewrite

The generator always contains one red marker in a fixed `3 x 5` grid. The old
model used two learned `3 x 3` convolutions with 191 parameters. The new model:

1. Uses a `1 x 1` convolution to encode every valid cell as `1` and the red
   marker as `2`.
2. Uses one analytically derived `3 x 3` convolution to produce background and
   the four diagonal colors while suppressing out-of-grid targets.
3. Removes the old ReLU and cuts the only intermediate tensor from 120 bytes to
   60 bytes.

The model passed the official train/test/ARC-GEN checks and all 15 legal marker
positions from the exact Google ARC-GEN generator.

## Files

- `model/task266.onnx`: accepted 832-byte model.
- `scripts/optimize_task266_analytic_marker_yusuke.py`: reproducible rewrite.
- `reports/score_report.json`: official local scoring result.
- `experiments/`: rejected task001 quantized outer-product attempt. Its flatten
  order is `(ab,xy)`, while the task requires Kronecker order `(ax,by)`.

## Next compression target

Cost 170 consists of 60 bytes of float intermediate memory and 110 parameters.
Crossing 20 requires cost 148 or lower. The remaining target is therefore a
22-element reduction in the final dense `10 x 1 x 3 x 3` route and bias without
creating another scored intermediate tensor.

The follow-up cross-20 search tested four smaller one-feature architectures with
estimated costs from 113 to 145. Exact linear programming rejected all 32
binary codes, 20,024 integer state codes, and 1,944 dilation/alignment layouts.
Continuous ReLU searches converged to fixed boundary-conflict sets across many
random starts. See `reports/cross20_structure_search.json`. Do not spend more
training time on a one-feature `2 x 2` bottleneck; the remaining solution must
change the output representation or exploit a different operator family.

The same pass tested a rank-3 output-direct decomposition for `task313`. It
would have cost 135 and crossed 20, but both random and SVD-initialized searches
retained thousands of sign errors. The fourth basis is required to represent
the generator's simultaneous period-2 and period-3 phases. Details are in
`reports/task313_rank3_search.json`.

Round 2 tested more aggressive operator changes on tasks 181, 285, and 395.
The task285 no-pad model gained `+0.053486` locally but failed on fresh exact
generator case 126 with Gather index 924, so it is explicitly rejected and was
not submitted. Dynamic task181 indices cost more memory than the existing static
ScatterND table, and ONNX type checking prevents task395 from concatenating a
sparse zero initializer with dense bool tensors. See
`reports/20260712_bold_structure_search_round2.json`.
