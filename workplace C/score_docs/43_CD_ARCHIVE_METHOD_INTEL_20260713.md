# C+D Archive Method Intel - 2026-07-13

## Source And Compliance

- Local attachment: `E:/archive.zip`
- SHA256: `77ed5c85d75e1ed885dc37e5d5b73c0f46a1aa8c4a996c6c41dbab5af3c4277a`
- Archive root: `submission7300+`
- ONNX files: `399`; missing `task173.onnx`
- Referenced dataset: `https://www.kaggle.com/datasets/zealous9230/neurogolf7300`
- Primary discussion: `https://www.kaggle.com/competitions/neurogolf-2026/discussion/724725`
- Rules concern: `https://www.kaggle.com/competitions/neurogolf-2026/discussion/724717`

The latest discussion does not describe an optimization algorithm. It links a
public 7300-level model archive while participants explicitly question whether
using it during the final seven days is compliant. No host clarification was
captured in the ingested thread. Therefore all attachment-derived candidates
are marked `compliance_hold=true` and `do_not_submit`.

## Full Benchmark

The official local utility evaluated the current 7278.75 parent and attachment
model on every public train/test/arc-gen example for all C+D tasks.

| metric | value |
| --- | ---: |
| C+D tasks | 134 |
| parent models valid | 134 |
| attachment models present and valid | 133 |
| missing attachment model | 1 (`task173`) |
| attachment lower than parent | 54 |
| lower tasks owned by C | 30 |
| lower tasks owned by D | 24 |

Directly adopting those 54 models would imply about `+35.4271` local points,
but direct adoption was deliberately not performed.

## Learned Structural Patterns

The strongest recurring pattern is direct-output tensor contraction. Eight of
the 54 lower-cost models are a single `Einsum`; six replace another single
`Einsum`, and two replace a single `Conv`. Larger wins also use:

1. Rule-level graph replacement instead of local metadata cleanup.
2. Quantized local detection (`QLinearConv`) instead of long boolean chains.
3. Bitwise packed state for repeated masks and component propagation.
4. Fewer materialized spatial tensors, especially replacing
   Slice/Where/Scatter pipelines with direct-output contractions.
5. Shape-specific selectors embedded in one final operator.

Representative structure reductions include `task045 11->1` nodes,
`task099 56->1`, `task157 489->132`, `task251 26->11`, and
`task383 54->28`.

## Independent Method Transfer

The latest GitHub commit's B/task104 rank/sign analysis was applied without
using the attachment as a replacement:

- `task075`: exact rank-one template contraction was moved into a cropped 1x1
  Conv and marker extraction into a stride-cropped Conv. Cost `1487->1394`,
  validation `265/265`, delta points `+0.0645833551`.
- `task315`: a rank-two color factor works under argmax but fails the official
  per-channel `output > 0` decoder on `0/266`. It is rejected. This confirms
  that sign margin, not class ordering, is the required optimization target.

## One-Round Derived Optimization

All 133 attachment models were passed through our own deterministic round:

1. identical initializer deduplication;
2. unused initializer removal;
3. Conv/QLinearConv zero-support crop;
4. dilation-aware pad compensation;
5. shared-weight protection;
6. ONNX checker plus official full validation.

Six graphs changed. Four became cheaper than the attachment but still did not
beat the current parent. Two are lower than both source and parent, but remain
on compliance hold:

| task | owner | parent | attachment | derived | validation |
| --- | --- | ---: | ---: | ---: | ---: |
| task158 | C | 26250 | 18560 | 18530 | 266/266 |
| task182 | D | 6095 | 6095 | 6065 | 267/267 |

No attachment-derived ONNX is approved for submission.

## Artifacts

- Full benchmark: `41_CD_NEUROGOLF7300_BENCHMARK.csv`
- Benchmark summary: `41_CD_NEUROGOLF7300_BENCHMARK.json`
- Derived results: `42_CD_ARCHIVE_ONE_ROUND_RESULTS.csv`
- Derived summary: `42_CD_ARCHIVE_ONE_ROUND_RESULTS.json`
- Candidate registry: `44_CD_ARCHIVE_DERIVED_CANDIDATES.md`
- Scripts: `cd_archive_benchmark.py`, `cd_archive_one_round_optimizer.py`
