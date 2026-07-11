# task193 Analysis

- Public examples: 3 train, 1 test, 262 arc-gen, 266 total.
- Python rule: 266/266. A non-background cell survives only with at least two same-color orthogonal peers.
- The baseline Conv has 91 nonzero weights, but it is not fully depthwise. Outputs 1..9 are independent five-tap filters; output 0 has 45 cross-color negative weights.
- The old `group=10` replacement failed at borders because a zero-padded background channel cannot distinguish a corner from an isolated interior foreground cell.

## 2026-07-11 Exact Factor Probe

The candidate separated foreground channels, rebuilt output 0 from their sum, and reproduced all 266 outputs exactly. It also created full-grid intermediate tensors:

| artifact | valid | memory | params | cost | delta points |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | true | 0 | 910 | 910 | 0 |
| exact factor | true | 82,800 | 112 | 82,912 | -4.512090 |

Sparse initializer storage was also tested. ONNX checker rejects a sparse tensor as a `Conv` weight input, so it is not a valid submission format for this task.

Conclusion: preserve the one-node baseline for task193. Any next attempt must retain one-node execution while encoding the background boundary condition; branching is categorically cost-negative under the official memory scorer.
