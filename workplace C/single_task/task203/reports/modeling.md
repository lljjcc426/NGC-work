# task203 independent modeling

## Rule

The input is a set of concentric rectangular color rings. The output reverses the ring-color order: the center color becomes the outer ring color, the next inner color becomes the next outer color, and so on. Ring area makes each used color count unique. For every color count `c`, its partner count is `largest_count + 4 - c`.

## Candidate structure

`task203_frequency_involution.onnx` counts every channel, computes the complementary target count, constructs the full count-to-target equality matrix by explicit Unsqueeze broadcasting, obtains the involutive color permutation with ArgMax, and applies it with Gather. It independently encodes the ring-frequency rule and replaces the baseline target/transpose broadcast topology.

## Validation

- Public examples: 4 train, 1 test, 262 arc-gen.
- Candidate: 267/267 exact.
- Baseline cost: 355 (memory 352, params 3).
- Candidate cost: 795 (memory 788, params 7).
- Decision: rule accepted as an independent model; replacement rejected because explicit broadcast tensors cost more than the baseline transpose formulation.
