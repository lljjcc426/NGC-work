# Parallel Three-Task Build Result

No submission was created or sent.

| task | Python rule | ONNX examples | old cost | new cost | delta points | accepted |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| task193 | 266/266 | 266/266 | 910 | 82,912 | -4.512090 | false |
| task372 | 266/266 | 266/266 | 710 | 7,135 | -2.307503 | false |
| task332 | 267/267 | 267/267 | 561 | 6,150 | -2.394486 | false |

The common blocker is official activation-memory scoring: direct or factored ONNX graphs cost more than the original single-node models even when parameter counts decrease. Sparse Conv initializers are rejected by ONNX checker.

Next selection rule: prioritize tasks whose original graph can be compressed inside one operator. Do not pursue a multi-node rule rebuild unless its intermediate tensors stay smaller than the baseline activation footprint.
