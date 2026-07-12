# Final task383

- method: crop then color decode
- full validation: 266/266
- old cost: 5830
- new cost: 33497
- delta cost: 27667
- accepted: false
- artifact: `E:\kongming\NGC-work\workplace C\single_task\task383\onnx\task383_crop_then_color_decode.onnx`
- next action: retain as a falsified cost hypothesis and redesign

## Accepted Baseline Compression

The earlier Conv/crop collapse remains the best valid model: `5830 -> 5800`,
validated 266/266.

## Activation Crop Boundary Probe

The apparent inactive tail cannot be removed safely:

| crop | nominal cost | validation |
| --- | ---: | ---: |
| 23x24 | 5642 | 257/266 |
| 24x23 | 5640 | 264/266 |
| 23x22 | 5286 | 242/266 |

The tail carries color-10 boundary-sentinel semantics even when the ordinary
activation mask is zero. All three activation-crop variants are rejected.
