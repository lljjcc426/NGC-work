# FINAL TASK075 REPORT

- Rule model: copy the upper-left `3x3` color template at positions selected by
  the `3x3` marker lattice
- Accepted structure: rank-one color crop Conv plus strided marker crop Conv
- Official validation: `265/265`
- Baseline memory / params / cost: `1326 / 161 / 1487`
- Candidate memory / params / cost: `1326 / 68 / 1394`
- Delta cost: `-93`
- Baseline points: `17.6954840535`
- Candidate points: `17.7600674087`
- Delta points: `+0.0645833551`
- Status: accepted local replacement
- Artifact: `workplace C/single_task/task075/onnx/task075_candidate.onnx`
- Builder: `workplace C/single_task/task075/scripts/build_rank1_crop_conv.py`

The original `templ_scalar_f` coefficient tensor is already exact rank one.
Unlike B/task104, this task needs exact ten-color identity rather than a binary
complementary sign mask, so a rank-two sign decoder does not provide a smaller
end-to-end representation. The accepted builder instead maps that rank-one
contraction to a cropped `1x1 Conv`, eliminating the two spatial selector
tensors without materializing the one-hot template. Marker extraction is
similarly fused into a cropped stride-three `1x1 Conv`.

The prior Slice-plus-ArgMax candidate remains rejected at cost `1788`.
`reports/sign_rank_evidence.csv` and `scripts/analyze_sign_rank.py` preserve
the reproducible rank and decoder lower-bound evidence.
