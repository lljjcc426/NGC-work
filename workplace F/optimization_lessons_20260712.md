# F Workplace Optimization Lessons Through v93

Updated: 2026-07-12 (Asia/Shanghai)

This note summarizes the verified optimization experience accumulated after the
v81 F checkpoint.  It is an engineering handoff, not a claim that the F goal is
finished.  The current official base is v93 at public score `7273.37`; 50 of the
67 F/shared tasks are still below 20 points.

## 1. Current verified state

- Current official package: `v93_ngc_remote_verified_merge`.
- Kaggle submission ref: `54592484`.
- Public score: `7273.37`.
- Package SHA256:
  `d3284267c02846dde8571890d4c761dcf9592fce2ec190c3348a0dee1c13c44f`.
- F/shared task count: 67.
- F/shared tasks currently below 20: 50.
- F tasks promoted above 20 by dedicated rewrites so far:
  `task006`, `task026`, `task274`, `task306`, and `task334`.
- The current objective remains: promote every remaining F/shared task to at
  least 20 under the official verifier.

## 2. Score geometry: design for cost <= 148

The official score is approximately:

`points = 25 - ln(memory_bytes + parameter_elements)`.

Therefore, crossing 20 points requires total scored cost at or below roughly
148.  Small percentage reductions on a large model improve the leaderboard but
do not solve the F 20+ objective.  For target promotion, architecture selection
should start from a credible path to `cost <= 148`.

The scorer has several consequences that strongly affect design:

1. Initializers are charged by element count, not byte count.  Converting a
   float32 weight initializer to int8 does not reduce parameter cost unless the
   number of elements also falls.
2. Intermediate tensors are charged by bytes and summed across graph values.
   Reducing an intermediate from float32 to bool/uint8/int8 is often valuable.
3. The graph input and final graph output are not charged as intermediate
   memory.  A large operation is most attractive when it writes directly to
   `output`.
4. A new Cast can erase the benefit of quantization because both its source and
   destination intermediates are charged.
5. Scalar attributes are often free, while repeated tensor controls and vector
   attributes may add parameter elements.

## 3. Highest-value successful patterns

### 3.1 Replace learned dense logic with exact task algebra

The largest F promotions came from understanding the ARC rule and rebuilding it
with a compact exact expression instead of compressing the existing graph.

- `task026`: signed two-stage Conv, cost `200 -> 120`, points
  `19.701683 -> 20.212508`.
- `task274`: compact black/red representation plus grouped QLinearConv, cost
  `175 -> 117`, points `19.835214 -> 20.237826`.
- `task334`: compact black/gray representation plus grouped QLinearConv, cost
  `194 -> 121`, points `19.732142 -> 20.204209`.
- `task306`: replaced a 300-parameter periodic tiling network with one low-rank
  high-order Einsum, cost `300 -> 146`, points
  `19.296218 -> 20.016393`.

General lesson: when a model is above cost 300, algebraic rule recovery is more
promising than deleting a few controls.

### 3.2 Use signed margins instead of full masks

`task021` was reduced from cost `324` to `227` by replacing float16 spatial
masks with int8 signed row/column margins and one variadic `Min`.  It passed all
official examples and raised points from `19.219256` to `19.575050`.

The reusable pattern is:

- compute small scalar dimensions/counts;
- subtract compact coordinate thresholds;
- keep the margins signed so values outside the wanted region are non-positive;
- combine broadcast margins with `Min` or `Max` directly into the final output.

This is a strong compression pattern, but task021 still needs a fundamentally
smaller coordinate construction to cross 20.

### 3.3 Compress dynamic index generation with modular CRT seeds

`task249` was reduced from cost `261` to `203` using a compact modular seed for
its dynamic Gather indices.  `task231` was reduced from cost `219` to `179` by
replacing a Less/Where/Concat index path with a Pad/Reduce/Add/Cast/CRT-Mod path.

For a small finite set of possible widths, one integer seed can encode different
required residues under different moduli.  This removes branch tensors and
large condition masks.

Important limit: Gather indices must be int32 or int64 under the official ONNX
checker.  A dynamic 30-element int32 index tensor alone costs 120 bytes.  For
task231 and task249, this is now the main barrier to `cost <= 148`.

### 3.4 Remove representational overhead before changing semantics

Several global checkpoints accumulated small, safe gains by removing ONNX
overhead while preserving the exact computation:

- v85: move constant controls into equivalent legacy attributes and remove
  explicit defaults.
- v86: collapse broadcast-only initializer axes.
- v87: remove initializer values equal to ONNX schema defaults.
- v88: replace uniform Concat padding with Pad.
- v89: collapse broadcast-only Einsum constants and compact zero tails.
- v90: restrict Pad-18 control vectors to only affected axes.
- v91: merge seven additional independently verified deterministic reductions.

These passes are worth running globally because they are low-risk and additive,
but they rarely move a hard F task across 20 by themselves.

### 3.5 Rescore public models against the current parent

Do not trust a candidate's reported delta against an older package.  Extract the
candidate ONNX, compare its bytes to the current parent, then run the complete
official verifier task by task.

This procedure produced v93 from the updated public `NGC-work` remote state:

- 21 unique variants were independently scored against v92.
- Nine replacements were accepted:
  `task029`, `task205`, `task209`, `task233`, `task255`, `task277`, `task328`,
  `task368`, and `task377`.
- Local delta: `+1.0806035366`.
- Official score moved from `7272.29` to `7273.37`, exactly consistent after
  leaderboard rounding.

These nine tasks improve the shared global parent but do not reduce the current
50-task F below-20 queue.

## 4. Validation findings that prevent wasted work

### 4.1 Logical output trimming is not verifier-equivalent

A tempting task231/task249 rewrite shortened the model output width to the
maximum logical ARC width.  Decoding the shortened tensor produced the exact
expected grids, and the estimated costs were excellent (`129` for task231 and
`90` for task249).

It nevertheless failed every official example because the verifier first
compares the raw network tensor with the fixed `[1, 10, 30, 30]` benchmark using
`numpy.array_equal`.  A smaller output shape is rejected even when
`convert_from_numpy` would decode the same grid.

Rule: every accepted model must retain the full fixed output tensor shape.  Do
not use decoded-grid equality as a substitute for the official verifier.

### 4.2 Sparse Conv weights are not supported by the runtime path

Sparse initializers looked ideal for tasks such as task272 and task352 because
their dense Conv tensors contain very few nonzero values.  Full checker/runtime
experiments rejected sparse Conv weights.  A Constant with `sparse_value` merely
materializes a dense intermediate and loses the scoring benefit.

### 4.3 Integer operator support is operator-specific

- QLinearConv is useful when the input can remain compactly quantized and the
  operation writes close to the final output.
- The current runtime does not implement the tested int8 Einsum and int8 OneHot
  paths used by task253/task290 probes.
- Quantizing a graph blindly is counterproductive when it adds a large Cast of
  the full 30x30x10 input.

Always test checker support, ORT session creation, complete examples, and scored
memory separately.

### 4.4 Fixed linear filters cannot replace every conditional rule

For task272, a proposed grouped-Conv rewrite tried to infer isolated red cells
from the black channel alone.  A strict linear-program feasibility check over
all official 3x3 patches, including padded borders and corners, was infeasible.
The grid exterior and red cells are both zero in that channel, so boundary cases
cannot be separated by one fixed black-only filter.

This negative result prevents repeatedly trying minor coefficient variants of
the same impossible model family.

### 4.5 Training probes need a structural lower-bound story

Repeated learned low-rank or separable approximations are not useful merely
because they reduce MSE.  NeuroGolf requires exact sign behavior on every cell.
Before a long fit, establish that the proposed rank, grouping, receptive field,
and dtype can represent the rule and can plausibly fit under cost 148.

## 5. Current hard-task priorities

### Tier A: closest to 20

- `task231`: cost `179`, points `19.812614`; dynamic int32 Gather index is the
  dominant lower bound.
- `task249`: cost `203`, points `19.686794`; compact uint8 index construction is
  already used, but the final int32 Cast and full 30-index output remain costly.
- `task021`: cost `227`, points `19.575050`; signed-margin rewrite is valid, but
  row/column coordinate thresholds still dominate parameters.

These tasks need operator fusion or a direct final-output construction, not
another minor initializer cleanup.

### Tier B: rules that may admit an exact new architecture

- `task272`: isolated red becomes blue; connected red remains red.  Current
  dense one-node Conv is exact, but sparse Conv is unavailable and simple
  black-only grouped Conv is provably insufficient on borders.
- `task171`: draw an azure border around an all-black rectangle.  The rule is
  simple, but mapping one input channel to both output color 0 and color 8
  compactly, without a charged full-size intermediate, is the key obstacle.
- `task352`: exact one-node grouped Conv with many zeros; requires a supported
  factorization or a new algebraic rule, not sparse Conv.

## 6. Recommended operating loop

1. Start every target experiment from the latest verified parent, currently
   v93.
2. Derive the exact task rule and a cost lower bound before implementation.
3. Build one task-level ONNX candidate.
4. Run `onnx.checker.check_model(..., full_check=True)` and strict shape
   inference.
5. Run every train, test, and ARC-GEN example with the official verifier.
6. Require zero wrong examples and a real positive score delta.
7. For 20+ promotion, require candidate points `>= 20`, not merely an improved
   score.
8. Merge only accepted task bytes into the 400-file parent archive.
9. Validate exact names, uniqueness, CRC, and full ONNX checks for the package.
10. Submit to Kaggle, retrieve the official result, archive the receipt, and
    update the single current-base pointer.

## 7. Version progression after v81

| Version | Main change | Official public score |
| --- | --- | ---: |
| v81 | task006 and task306 promoted above 20 | 7270.18 |
| v82 | 16 fully verified public-package improvements | 7271.29 |
| v83 | task249 CRT/index cleanup plus deterministic pruning | 7271.55 |
| v84 | task231 compact CRT-Mod Gather index path | 7271.75 |
| v85 | legacy attributes/default cleanup | 7271.76 |
| v86 | broadcast-axis collapse | 7271.79 |
| v87 | optional-default removal | 7271.80 |
| v88 | Concat-fill to Pad rewrites | 7271.84 |
| v89 | Einsum broadcast and zero-tail collapse | 7271.90 |
| v90 | compact Pad axes | 7271.93 |
| v92 | task021 signed margins plus v91 deterministic gains | 7272.29 |
| v93 | nine independently rescored NGC-work remote gains | 7273.37 |

The leaderboard has improved substantially, but completion remains defined by
the per-task F requirement: all 67 F/shared tasks must individually score at
least 20.
