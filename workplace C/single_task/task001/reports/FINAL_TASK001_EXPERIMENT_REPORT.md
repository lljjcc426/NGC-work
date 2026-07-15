# task001 cost-below-190 investigation

## Result

No checker-valid, exact model below the round36 parent cost of 190 was found.
The parent remains the only tested model that is both below cost 4590 and exact.
No Kaggle submission was made and no shared script was changed.

The experiment is reproducible with:

```powershell
python "workplace C\single_task\task001\scripts\experiment_task001.py" --exhaustive
```

The command checks all 268 bundled examples and all 4608 monochrome 3x3 states.
Machine-readable results are in `task001_experiment_results.json`.

## Measured models

| model | memory | params | cost | bundled | all states | SHA256 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| round36 parent single Einsum | 0 | 190 | 190 | 268/268 | 4608/4608 | `3b216bf603d1bf6bacdaa9bff5078321266904b8c4d94cd251c63df79ea109d8` |
| dilated dynamic ConvTranspose | 36 | 40 | 76 | 0/268 | 9/4608 | `396c2edb81c53b4c9ee0a8eda526300442fcba4e5ed0e800622f2a3e6446f840` |
| ConvTranspose, background bias +0.5 | 36 | 50 | 86 | 0/268 | not run | `4f922a645b7de2fd55536f861d409dd310efee636a9b51ce6e400f7952ddebf2` |
| ConvTranspose, background bias -0.5 | 36 | 50 | 86 | 0/268 | not run | `e7704fcca460a3d67ce8132f881743765ff54b9684dd00403fd1700a9887da33` |
| best tested rank-1 Einsum | 0 | 100 | 100 | 23/268 | 297/4608 | `6808225249aff2395fae747928933de03c810ba2447f309af6db253c6603f3b5` |
| exact two-channel dynamic ConvTranspose + Pad | 4500 | 90 | 4590 | 268/268 | 4608/4608 | `2f5215b092decc40b5d7ee0cba2df0a20adca1ffdca683ebf635fc9d99835b66` |
| two-channel ConvTranspose with output_shape 30x30 | 1260 | 90 | 1350 | 0/268 | not run | `a5aad6f526e6a61cd7afbf6cb88b8b9696dda0d2298b3885932d2fd256e43ec4` |
| sparse parent selector | unscoreable | theoretical 37 | unscoreable | ORT 268/268 | not run | `8f6d5297fd7f1381ce2951dede592f3852c3823ed28bcfb3f72e06ac408b45cc` |

## ConvTranspose finding

The cost-76 graph computes every foreground channel exactly on all 268 bundled
examples. Only channel 0 is wrong, with 10,071 mismatched bundled cells.

For macro foreground bit `A`, micro foreground bit `P`, and scalar background
bias `b`, its background score is:

`A * (1 - P) + b`

The macro-background/micro-foreground state requires `b > 0`, while the
macro-foreground/micro-foreground state requires `b <= 0`. Therefore no scalar
ConvTranspose bias can repair the background. A positive bias also activates
channel 0 outside the valid 9x9 output region.

The exact two-channel construction confirms the missing rank: one channel
stamps an all-background kernel for macro-background cells, while the other
stamps the dynamic one-hot sprite for macro-foreground cells. It is exact, but
the generated `[2,10,3,3]` dynamic kernel alone costs 720 bytes. The full safe
graph costs 4590. Forcing `output_shape=[30,30]` removes the 9x9 carrier but does
not mean zero-padding; it changes ConvTranspose geometry and fails every bundled
example.

## Einsum finding

The exact spatial coefficient tensor is

`T = macro_selector (x) macro_selector + micro_selector (x) micro_selector`.

Flattening the row and column relation gives rank 2 because the macro and micro
selectors are linearly independent. The parent stores exactly these two modes
in `m[2,30,3]`, costing 180 dense elements, plus the 10-element color vector.

Collapsing the two parent modes to `m[1,30,3]` lowers cost to 100 but is not
exact. A sweep of `m0 + alpha*m1` for alpha in
`{-4,-2,-1,-0.5,0,0.5,1,2,4}` found a best result of only 23/268 bundled and
297/4608 exhaustive states. Thus the tested dense rank-1 family cannot replace
the rank-2 parent.

Sparsifying the exact parent selector leaves only 27 nonzero selector values,
for a theoretical cost of `27 + 10 = 37`. The saved debug graph runs 268/268
directly in ORT. It is not scoreable: both `check_model(full_check=True)` and
`infer_shapes(strict_mode=True)` reject the sparse Einsum operand as rank 0.
The pre-named sparse initializer bypasses the sanitizer naming bug, after which
the official-style scorer reaches and records the same strict inference error.
See `task001_sparse_parent_probe.json` for the exact error text.

## Conclusion

- ConvTranspose plus a constant bias cannot repair channel 0 by truth-table contradiction.
- An exact two-channel dynamic kernel exists, but its mandatory carriers exceed 190 before final output composition.
- The direct `output_shape=30x30` shortcut is not padding and is semantically wrong.
- The dense single-Einsum relation needs two selector modes; the tested rank-1 cost-100 family fails broadly.
- The exact theoretical cost-37 sparse form is rejected by the current mandatory strict checker.

Within these checker-valid dense families, cost 190 is the verified frontier.
