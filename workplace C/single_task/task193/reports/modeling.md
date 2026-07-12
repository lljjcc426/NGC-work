# task193 Independent Model

## Rule

The input contains solid same-color motifs plus sparse cells of that color. A
foreground cell is retained exactly when it has at least two orthogonally
adjacent cells of the same color. Other foreground cells are erased to color 0.
The output keeps the input shape inside the fixed 30x30 tensor support.

## Failed 3x3 factorization

The old dense `Conv` spends 900 weights because output channel 0 reads every
input color. A first `group=10` rewrite used a constant background logit. Its
cropped ARC grids were correct, but it incorrectly emitted background outside
the valid rectangle, so official one-hot validation failed `0/266`.

A 3x3 background-only classifier is provably infeasible on the public corpus:
the same 3x3 channel-0 patch occurs with both positive and negative output-0
labels at grid boundaries.

## Accepted 4x4 model

The first valid graph used a 5x5 depthwise kernel and cost 260. A complete
contiguous-support search then found that the asymmetric window from offsets
`[-2,+1]` on both axes is sufficient. The final graph is one `group=10 Conv`
with a 4x4 kernel and pads `[2,2,1,1]`:

- channel 0 uses a hard-margin linear support/noise classifier over the 4x4
  background-channel patch;
- channels 1..9 use independently fitted hard-margin local classifiers;
- a foreground logit is positive exactly for center-present cells with at least
  two same-color orthogonal neighbors;
- there are no intermediate tensors, branches, masks, or post-processing ops.

The background kernel was fitted as one shared linear rule over local patches;
it does not store or select complete sample outputs.

## Official Result

| artifact | nodes | memory | params | cost | validation |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline dense Conv | 1 | 0 | 910 | 910 | 266/266 |
| 5x5 depthwise threshold | 1 | 0 | 260 | 260 | 266/266 |
| 4x4 depthwise hard-margin | 1 | 0 | 170 | 170 | 266/266 |

Cost delta is `-740`; points improve from `18.1865554005` to
`19.8642015629`, an expected gain of `+1.6776461625`.

The 3x3, 3x4, and 4x3 background constraint systems are infeasible, making 4x4
the smallest contiguous centered support found for the one-node depthwise form.
