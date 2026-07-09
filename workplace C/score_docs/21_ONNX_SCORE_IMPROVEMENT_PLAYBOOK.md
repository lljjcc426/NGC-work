# ONNX Score Improvement Playbook

Generated: 2026-07-09T15:58:38

Current generic cleanup result: no cost decrease on P0/P1. Future score work should avoid repeating optimizer/sim-only passes unless the source graph changes.

Highest-value tactics:

1. Dedicated compact builder for `task158`: detect motif/template and emit smaller same-shape fill network.
2. Dedicated compact builder for `task286`: replace 2393-node iterative graph with a compact propagation primitive if the color-fill rule is formalized.
3. Dedicated compact builder for `task054`: line/cross propagation from marker pixel, not generic graph simplification.
4. Dedicated compact builder for `task364`: classify connected shape glyphs into colors 1/2/6 with a component/neighborhood network.
5. Mine `prvsiyan_7266_72/output/visualizations` for task-level explanations before coding more ONNX.
