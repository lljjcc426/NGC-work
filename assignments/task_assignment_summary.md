# NeuroGolf 400 Task Assignment
Generated: 2026-07-09

## Evidence
- Main score/cost source: F:/kaggle/neurogolf-2026/scripts/v244_costs.txt, 397 parsed rows.
- Missing from v244_costs: task067, task179, task241. Filled from F:/kaggle/neurogolf-2026/reports/v241_deadnode_manifest_20260708.csv and marked P0_check_score_anomaly.
- Structure features came from F:/kaggle/neurogolf-2026/data/task001.json through task400.json.
- Known high-value/high-risk notes came from F:/kaggle/neurogolf-2026/reports/high_cost_task_triage.md.

## Assignment Rule
- All 400 unique tasks have one primary owner.
- Per request, every member has 67 assignment slots. Because 6*67=402, there are 2 shared_review duplicate slots.
- Shared review duplicates: task233 -> E, task366 -> F. Primary owner for both remains A.
- To make every member exactly 67 slots, task337 was moved from E to F.
- CSV source of truth: assignments/task_assignment_400.csv. Use assignment_type to distinguish primary from shared_review.

## Member Summary
| owner | track | slots | unique_tasks | shared | gap_to_18_sum | cost_sum | anomaly | P0_lt16 | P1_16_16p7 | P2_16p7_17p5 | P3_ge17p5 | shape_change | mixed_shape | same_shape |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | shape_relayout_high_cost | 67 | 67 | 0 | 43.486 | 225348 | 0 | 6 | 8 | 13 | 40 | 23 | 0 | 44 |
| B | same_shape_rules_fill | 67 | 67 | 0 | 42.869 | 198367 | 0 | 7 | 7 | 13 | 40 | 26 | 0 | 41 |
| C | onnx_equiv_compression | 67 | 67 | 0 | 43.868 | 230105 | 0 | 6 | 8 | 13 | 40 | 25 | 0 | 42 |
| D | mid_low_score_rule_mining | 67 | 67 | 0 | 42.868 | 194710 | 0 | 7 | 8 | 12 | 40 | 15 | 0 | 52 |
| E | public_source_ab_testing | 67 | 67 | 1 | 45.941 | 216905 | 0 | 8 | 8 | 13 | 38 | 22 | 0 | 45 |
| F | tail_validation_packaging | 67 | 67 | 1 | 86.641 | 169833 | 3 | 6 | 5 | 13 | 40 | 28 | 1 | 38 |

## Top 12 Per Member

### Member A: shape_relayout_high_cost
| task | type | priority | points | cost | gap_to_18 | shape | size | color | note |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| task233 | primary | P0_lt16 | 14.554188 | 34400 | 3.445812 | shape_change | shrink | input_palette_only |  |
| task366 | primary | P0_lt16 | 14.573442 | 33744 | 3.426558 | shape_change | shrink | input_palette_only |  |
| task367 | primary | P0_lt16 | 15.277615 | 16687 | 2.722385 | same_shape | same_size | new_output_colors |  |
| task187 | primary | P0_lt16 | 15.332868 | 15790 | 2.667132 | same_shape | same_size | new_output_colors |  |
| task379 | primary | P0_lt16 | 15.884520 | 9095 | 2.115480 | same_shape | same_size | input_palette_only |  |
| task338 | primary | P0_lt16 | 15.912053 | 8848 | 2.087947 | same_shape | same_size | new_output_colors |  |
| task202 | primary | P1_16_16p7 | 16.037608 | 7804 | 1.962392 | same_shape | same_size | input_palette_only |  |
| task005 | primary | P1_16_16p7 | 16.181370 | 6759 | 1.818630 | same_shape | same_size | input_palette_only |  |
| task017 | primary | P1_16_16p7 | 16.227700 | 6453 | 1.772300 | same_shape | same_size | input_palette_only |  |
| task319 | primary | P1_16_16p7 | 16.328713 | 5833 | 1.671287 | shape_change | shrink | input_palette_only |  |
| task004 | primary | P1_16_16p7 | 16.459090 | 5120 | 1.540910 | same_shape | same_size | input_palette_only |  |
| task148 | primary | P1_16_16p7 | 16.505666 | 4887 | 1.494334 | same_shape | same_size | new_output_colors |  |

### Member B: same_shape_rules_fill
| task | type | priority | points | cost | gap_to_18 | shape | size | color | note |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| task018 | primary | P0_lt16 | 14.899302 | 24360 | 3.100698 | same_shape | same_size | input_palette_only |  |
| task285 | primary | P0_lt16 | 15.111626 | 19700 | 2.888374 | same_shape | same_size | input_palette_only |  |
| task101 | primary | P0_lt16 | 15.473026 | 13725 | 2.526974 | same_shape | same_size | input_palette_only |  |
| task076 | primary | P0_lt16 | 15.540848 | 12825 | 2.459152 | same_shape | same_size | input_palette_only |  |
| task350 | primary | P0_lt16 | 15.891028 | 9036 | 2.108972 | same_shape | same_size | new_output_colors |  |
| task255 | primary | P0_lt16 | 15.904958 | 8911 | 2.095042 | same_shape | same_size | new_output_colors |  |
| task023 | primary | P0_lt16 | 15.985431 | 8222 | 2.014569 | same_shape | same_size | new_output_colors |  |
| task209 | primary | P1_16_16p7 | 16.041331 | 7775 | 1.958669 | shape_change | shrink | input_palette_only |  |
| task328 | primary | P1_16_16p7 | 16.200489 | 6631 | 1.799511 | same_shape | same_size | input_palette_only |  |
| task280 | primary | P1_16_16p7 | 16.295166 | 6032 | 1.704834 | same_shape | same_size | input_palette_only |  |
| task368 | primary | P1_16_16p7 | 16.457139 | 5130 | 1.542861 | same_shape | same_size | input_palette_only |  |
| task205 | primary | P1_16_16p7 | 16.504848 | 4891 | 1.495152 | shape_change | shrink | input_palette_only |  |

### Member C: onnx_equiv_compression
| task | type | priority | points | cost | gap_to_18 | shape | size | color | note |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| task158 | primary | P0_lt16 | 14.742937 | 28483 | 3.257063 | same_shape | same_size | input_palette_only |  |
| task286 | primary | P0_lt16 | 14.799784 | 26909 | 3.200216 | same_shape | same_size | input_palette_only |  |
| task054 | primary | P0_lt16 | 14.857732 | 25394 | 3.142268 | same_shape | same_size | input_palette_only |  |
| task364 | primary | P0_lt16 | 15.057196 | 20802 | 2.942804 | same_shape | same_size | new_output_colors |  |
| task349 | primary | P0_lt16 | 15.391421 | 14892 | 2.608579 | same_shape | same_size | new_output_colors |  |
| task077 | primary | P0_lt16 | 15.951003 | 8510 | 2.048997 | same_shape | same_size | new_output_colors |  |
| task096 | primary | P1_16_16p7 | 16.053886 | 7678 | 1.946114 | shape_change | shrink | input_palette_only |  |
| task009 | primary | P1_16_16p7 | 16.190286 | 6699 | 1.809714 | same_shape | same_size | input_palette_only |  |
| task383 | primary | P1_16_16p7 | 16.315261 | 5912 | 1.684739 | same_shape | same_size | input_palette_only |  |
| task382 | primary | P1_16_16p7 | 16.351428 | 5702 | 1.648572 | same_shape | same_size | input_palette_only |  |
| task278 | primary | P1_16_16p7 | 16.480809 | 5010 | 1.519191 | same_shape | same_size | new_output_colors |  |
| task165 | primary | P1_16_16p7 | 16.520716 | 4814 | 1.479284 | same_shape | same_size | input_palette_only |  |

### Member D: mid_low_score_rule_mining
| task | type | priority | points | cost | gap_to_18 | shape | size | color | note |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| task133 | primary | P0_lt16 | 15.034571 | 21278 | 2.965429 | same_shape | same_size | input_palette_only |  |
| task002 | primary | P0_lt16 | 15.433035 | 14285 | 2.566965 | same_shape | same_size | new_output_colors |  |
| task243 | primary | P0_lt16 | 15.478212 | 13654 | 2.521788 | same_shape | same_size | input_palette_only |  |
| task173 | primary | P0_lt16 | 15.498259 | 13383 | 2.501741 | same_shape | same_size | input_palette_only |  |
| task145 | primary | P0_lt16 | 15.538512 | 12855 | 2.461488 | same_shape | same_size | new_output_colors |  |
| task074 | primary | P0_lt16 | 15.889480 | 9050 | 2.110520 | same_shape | same_size | input_palette_only |  |
| task219 | primary | P0_lt16 | 15.924449 | 8739 | 2.075551 | same_shape | same_size | new_output_colors |  |
| task157 | primary | P1_16_16p7 | 16.168434 | 6847 | 1.831566 | same_shape | same_size | new_output_colors |  |
| task089 | primary | P1_16_16p7 | 16.184333 | 6739 | 1.815667 | same_shape | same_size | input_palette_only |  |
| task182 | primary | P1_16_16p7 | 16.284776 | 6095 | 1.715224 | same_shape | same_size | input_palette_only |  |
| task029 | primary | P1_16_16p7 | 16.426805 | 5288 | 1.573195 | shape_change | shrink | input_palette_only |  |
| task251 | primary | P1_16_16p7 | 16.498936 | 4920 | 1.501064 | same_shape | same_size | new_output_colors |  |

### Member E: public_source_ab_testing
| task | type | priority | points | cost | gap_to_18 | shape | size | color | note |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| task233 | shared_review | P0_lt16 | 14.554188 | 34400 | 3.445812 | shape_change | shrink | input_palette_only | shared review duplicate: high-cost shape-change task; pair with primary owner A for source/A-B evidence |
| task138 | primary | P0_lt16 | 15.526911 | 13005 | 2.473089 | shape_change | shrink | input_palette_only |  |
| task064 | primary | P0_lt16 | 15.551825 | 12685 | 2.448175 | same_shape | same_size | input_palette_only |  |
| task118 | primary | P0_lt16 | 15.584110 | 12282 | 2.415890 | same_shape | same_size | new_output_colors |  |
| task198 | primary | P0_lt16 | 15.657755 | 11410 | 2.342245 | same_shape | same_size | new_output_colors |  |
| task110 | primary | P0_lt16 | 15.734036 | 10572 | 2.265964 | same_shape | same_size | input_palette_only |  |
| task216 | primary | P0_lt16 | 15.881008 | 9127 | 2.118992 | shape_change | shrink | input_palette_only |  |
| task324 | primary | P0_lt16 | 15.907318 | 8890 | 2.092682 | same_shape | same_size | input_palette_only |  |
| task370 | primary | P1_16_16p7 | 16.000504 | 8099 | 1.999496 | same_shape | same_size | input_palette_only |  |
| task192 | primary | P1_16_16p7 | 16.170481 | 6833 | 1.829519 | same_shape | same_size | input_palette_only |  |
| task092 | primary | P1_16_16p7 | 16.212322 | 6553 | 1.787678 | same_shape | same_size | input_palette_only |  |
| task085 | primary | P1_16_16p7 | 16.409370 | 5381 | 1.590630 | same_shape | same_size | input_palette_only |  |

### Member F: tail_validation_packaging
| task | type | priority | points | cost | gap_to_18 | shape | size | color | note |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| task366 | shared_review | P0_lt16 | 14.573442 | 33744 | 3.426558 | shape_change | shrink | input_palette_only | shared review duplicate: high-cost shape-change task; pair with primary owner A for validation/package evidence |
| task067 | primary | P0_check_score_anomaly | 1.000000 | 0 | 17.000000 | shape_change | shrink | input_palette_only | score anomaly: absent from v244_costs; latest zip contains tiny ONNX; verify cost/points manually |
| task179 | primary | P0_check_score_anomaly | 1.000000 | 0 | 17.000000 | same_shape | same_size | input_palette_only | score anomaly: absent from v244_costs; latest zip contains tiny ONNX; verify cost/points manually |
| task241 | primary | P0_check_score_anomaly | 1.000000 | 0 | 17.000000 | same_shape | same_size | input_palette_only | score anomaly: absent from v244_costs; latest zip contains tiny ONNX; verify cost/points manually |
| task191 | primary | P0_lt16 | 15.686923 | 11082 | 2.313077 | same_shape | same_size | input_palette_only |  |
| task066 | primary | P0_lt16 | 15.717990 | 10743 | 2.282010 | same_shape | same_size | input_palette_only |  |
| task025 | primary | P0_lt16 | 15.766334 | 10236 | 2.233666 | same_shape | same_size | input_palette_only |  |
| task204 | primary | P0_lt16 | 15.766725 | 10232 | 2.233275 | same_shape | same_size | new_output_colors |  |
| task080 | primary | P0_lt16 | 15.812621 | 9773 | 2.187379 | same_shape | same_size | input_palette_only |  |
| task279 | primary | P1_16_16p7 | 16.377906 | 5553 | 1.622094 | same_shape | same_size | new_output_colors |  |
| task387 | primary | P1_16_16p7 | 16.481807 | 5005 | 1.518193 | same_shape | same_size | new_output_colors |  |
| task361 | primary | P1_16_16p7 | 16.522588 | 4805 | 1.477412 | same_shape | same_size | input_palette_only |  |

## Execution Rules
- Work on owned tasks first. Shared_review tasks must sync conclusions with the primary owner.
- For every single-task replacement, record source, build script, local points/cost, validation scope, and Kaggle A/B result if submitted.
- Adopt only changes with positive Kaggle publicScore evidence or clear reproducible validation evidence.
- Do not delete files silently. If cleanup is needed, record full path, size, reason, and time.
