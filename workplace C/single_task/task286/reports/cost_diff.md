# task286 Cost Diff

Baseline:

- artifact: `E:\kagglegolf\submissions\candidates\GOLF_20260709_101_prvsiyan_7266_72_repro\onnx\task286.onnx`
- old_cost: `26909`
- old_points: `14.799783917876258`
- examples_checked: `265`
- local_valid: `true`

Candidate status:

- no accepted ONNX candidate in this pass
- sparse-initializer probes failed ONNX checker/type inference before official cost validation
- baseline self-check result: `26909 -> 26909`

Probe summary:

| probe | potential saved params | result |
| --- | ---: | --- |
| sparse `em`, `d_Wci`, `pk_w*`, pad constants | 419 | rejected: Conv sparse weight unsupported |
| sparse `em`, `pk_w*`, pad constants | 389 | rejected: Where sparse condition unsupported |
| sparse `pk_w*`, pad constants | 77 | rejected: Pad sparse pads unsupported |
| sparse `pk_w*` only | 75 | rejected: MatMulInteger sparse input unsupported |

Conclusion: `task286` needs a rewritten lower-memory bitset flood-fill graph to improve cost. Constant-only surgery is not enough and is not checker-compatible for the relevant operators.
