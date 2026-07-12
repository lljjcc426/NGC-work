# FINAL task146 report

- Rule model: select the unique non-transpose-symmetric 3x3 tile.
- Baseline: one dense `10x3x3` checksum Conv followed by compact row selection.
- Round 2 candidate: factor the checksum into a `1x1` color-code Conv and a
  `3x3` spatial checksum Conv using collision-free coefficients `(1, 2, 4)`.
- Official validation: baseline `267/267`; round 2 candidate `267/267`.
- Official cost: baseline `265` (`memory=165`, `params=100`); candidate `302`
  (`memory=273`, `params=29`); delta `+37` cost / `-0.1306971914` points.
- Artifact: `onnx/task146_candidate.onnx`.
- Replacement accepted: no.
- Exact blocker: factorization removes 71 parameters but materializes the local
  `9x3` scalar-color tensor, adding 108 bytes of scored activation memory. The
  memory increase is larger than the parameter saving.
- Support lower bound on the 801 public tiles: every fixed rectangular linear
  checksum smaller than `3x3` leaves public asymmetric tiles inseparable from
  the span of symmetric tiles. The full `3x3` support is the first with zero
  such misses, so simply shrinking the baseline kernel cannot remain exact.
- Decision: keep the cost-265 baseline; stop this search family.
