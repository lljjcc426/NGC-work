# FINAL TASK230 REPORT

- Rule understood: decorate every 2x2 color-5 block with four fixed-color diagonal corner markers.
- Accepted structure: one `3x3 group=2 Conv` fitted with hard-margin constraints over every public input window.
- Full validation: 266/266 exact.
- Baseline cost: 900.
- Candidate cost: 460.
- Delta cost: -440.
- Baseline points: 18.197605236675905.
- Candidate points: 18.86877351051686.
- Delta points: +0.6711682738409545.
- Candidate: `onnx/task230_candidate.onnx`.
- Accepted replacement: yes.

The earlier pool/deconvolution rule model remains useful as a semantic proof but
cost 70,828 because of full-grid intermediates. The accepted model instead
keeps the scorer-optimal one-node form. A `group=2` split is the highest valid
grouping: all ten output channels are linearly separable in their corresponding
five-channel 3x3 windows, while `group=5` and `group=10` fail public constraints.
