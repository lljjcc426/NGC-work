# C Local Task Score Campaign - 2026-07-12

## Scope

This pass deliberately ignored parent packages, kernels, and Kaggle submission.
The only objective was to lower official local task cost with full public
train/test/arc-gen validation.

## Accepted Results

| task | structure | old cost | new cost | saved | points gain | checked |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| task193 | one-node 4x4 group=10 Conv | 910 | 170 | 740 | +1.677646 | 266/266 |
| task230 | one-node 3x3 group=2 Conv | 900 | 460 | 440 | +0.671168 | 266/266 |
| task372 | one-node 7x1 group=2 Conv | 710 | 360 | 350 | +0.679161 | 266/266 |
| task349 | width/halo activation crop | 14892 | 14647 | 245 | +0.016589 | 267/267 |
| task335 | exact rank-4 Einsum template | 1380 | 1324 | 56 | +0.041426 | 266/266 |
| task286 | zero-support Conv crop | 26909 | 26879 | 30 | +0.001115 | 265/265 |
| task069 | Pad compaction plus Conv crop | 2946 | 2916 | 30 | +0.010236 | 264/264 |
| task201 | Pad compaction plus Conv crop | 3043 | 3013 | 30 | +0.009908 | 266/266 |

Total expected local gain is `+3.107248641435721` points.

## Main Technical Finding

The official scorer thresholds every output channel independently at zero; it
does not use argmax. Therefore common-mode logit subtraction is invalid even
when it preserves the winning class. The productive replacement is a
hard-margin classifier over the finite set of public one-hot input windows,
while retaining a one-node Conv so scored intermediate memory remains zero.

This produced the three largest changes:

- task193: dense 3x3 Conv to asymmetric 4x4 depthwise Conv.
- task230: dense 3x3 Conv to group=2 3x3 Conv.
- task372: dense 7x1 Conv to group=2 7x1 Conv.

The fitted models are local rule classifiers, not complete-output lookup tables.
Each builder deduplicates local windows and enforces positive/negative margins
for every cell in every public example.

## Rejected Experiments

- task391 direct OneHot: 267/267 but cost 159 to 275; rejected.
- task372 common-mode grouped Conv: preserves argmax but fails independent
  threshold semantics, 0/266.
- task278 dense 5x5 Conv: channels 0 and 3 are not linearly separable.
- task278 group=2 7x7 Conv: channels 0 and 3 remain infeasible.
- task193 3x3, 3x4, and 4x3 depthwise supports: background constraints are
  infeasible; 4x4 is the smallest contiguous support found.
- task349 two-channel and rank-4 quantized halo variants: overlap cases create
  false positives; the accepted crop is smaller but exact.

## Next Local Targets

1. Apply finite-window grouped-Conv feasibility search to other parameter-only
   or low-memory classifiers before attempting graph expansion.
2. Rebuild task077 propagation with a wider single-step morphology operator;
   the two-round 5-wide model misses only 8/266 examples.
3. Search task349 overlap-safe channel codes with explicit collision constraints
   rather than polynomial feature guesses.
4. Rework task278 as a two-stage quantized nonlinear classifier; a single Conv
   has now been formally rejected for the relevant supports.
5. Factor task096's compact radius/code tables while preserving its existing
   low-width reduction path.

No submission-related action was performed.
