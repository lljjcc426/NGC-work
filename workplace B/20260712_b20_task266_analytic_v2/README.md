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
