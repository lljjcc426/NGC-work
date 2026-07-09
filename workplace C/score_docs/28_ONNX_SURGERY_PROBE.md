# C ONNX Surgery Probe

Generated: 2026-07-09T15:55:38
Tasks: `task158, task286, task054, task364, task349, task077, task096, task009, task383, task382, task278, task165, task378, task132`
Strategies: `optimizer, sim, optimizer_sim`
Output root: `E:\kongming\NGC-work\workplace C\artifacts\surgery_probes\20260709_155105`
Accepted improvements: `0`

| task | strategy | old_cost | new_cost | delta_cost | nodes | initializers | file_size | accepted | notes |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| task158 | baseline | 28483 | 28483 | 0 | 70->70 | 40->40 | 14407->14407 | False | `` |
| task158 | optimizer | 28483 | 28483 | 0 | 70->70 | 40->40 | 14407->14265 | False | `eliminate_deadend,eliminate_identity,eliminate_nop_cast,eliminate_nop_dropout,eliminate_nop_flatten,eliminate_nop_monoto` |
| task158 | sim | 28483 | 28483 | 0 | 70->70 | 40->40 | 14407->14265 | False | `onnxsim(input=1x10x30x30)` |
| task158 | optimizer_sim | 28483 | 28483 | 0 | 70->70 | 40->40 | 14407->14265 | False | `eliminate_deadend,eliminate_identity,eliminate_nop_cast,eliminate_nop_dropout,eliminate_nop_flatten,eliminate_nop_monoto` |
| task286 | baseline | 26909 | 26909 | 0 | 2393->2393 | 46->46 | 111759->111759 | False | `` |
| task286 | optimizer | 26909 | 26909 | 0 | 2393->2393 | 46->46 | 111759->111759 | False | `eliminate_deadend,eliminate_identity,eliminate_nop_cast,eliminate_nop_dropout,eliminate_nop_flatten,eliminate_nop_monoto` |
| task286 | sim | 26909 | 26909 | 0 | 2393->2393 | 46->46 | 111759->180771 | False | `onnxsim(input=1x10x30x30)` |
| task286 | optimizer_sim | 26909 | 26909 | 0 | 2393->2393 | 46->46 | 111759->180771 | False | `eliminate_deadend,eliminate_identity,eliminate_nop_cast,eliminate_nop_dropout,eliminate_nop_flatten,eliminate_nop_monoto` |
| task054 | baseline | 25394 | 25394 | 0 | 286->286 | 45->45 | 24729->24729 | False | `` |
| task054 | optimizer | 25394 | 25394 | 0 | 286->286 | 45->45 | 24729->24729 | False | `eliminate_deadend,eliminate_identity,eliminate_nop_cast,eliminate_nop_dropout,eliminate_nop_flatten,eliminate_nop_monoto` |
| task054 | sim | 25394 | 25394 | 0 | 286->286 | 45->45 | 24729->36065 | False | `onnxsim(input=1x10x30x30)` |
| task054 | optimizer_sim | 25394 | 25394 | 0 | 286->286 | 45->45 | 24729->36065 | False | `eliminate_deadend,eliminate_identity,eliminate_nop_cast,eliminate_nop_dropout,eliminate_nop_flatten,eliminate_nop_monoto` |
| task364 | baseline | 14642 | 14642 | 0 | 27->27 | 7->7 | 2168->2168 | False | `` |
| task364 | optimizer | 14642 | 14642 | 0 | 27->27 | 7->7 | 2168->2168 | False | `eliminate_deadend,eliminate_identity,eliminate_nop_cast,eliminate_nop_dropout,eliminate_nop_flatten,eliminate_nop_monoto` |
| task364 | sim | 14642 | 14642 | 0 | 27->27 | 7->7 | 2168->2227 | False | `onnxsim(input=1x10x30x30)` |
| task364 | optimizer_sim | 14642 | 14642 | 0 | 27->27 | 7->7 | 2168->2227 | False | `eliminate_deadend,eliminate_identity,eliminate_nop_cast,eliminate_nop_dropout,eliminate_nop_flatten,eliminate_nop_monoto` |
| task349 | baseline | 14892 | 14892 | 0 | 13->13 | 11->11 | 2664->2664 | False | `` |
| task349 | optimizer | 14892 | 14892 | 0 | 13->13 | 11->11 | 2664->2664 | False | `eliminate_deadend,eliminate_identity,eliminate_nop_cast,eliminate_nop_dropout,eliminate_nop_flatten,eliminate_nop_monoto` |
| task349 | sim | 14892 | 14892 | 0 | 13->13 | 11->11 | 2664->3089 | False | `onnxsim(input=1x10x30x30)` |
| task349 | optimizer_sim | 14892 | 14892 | 0 | 13->13 | 11->11 | 2664->3089 | False | `eliminate_deadend,eliminate_identity,eliminate_nop_cast,eliminate_nop_dropout,eliminate_nop_flatten,eliminate_nop_monoto` |
| task077 | baseline | 7657 | 7657 | 0 | 15->15 | 11->11 | 979->979 | False | `` |
| task077 | optimizer | 7657 | 7657 | 0 | 15->15 | 11->11 | 979->979 | False | `eliminate_deadend,eliminate_identity,eliminate_nop_cast,eliminate_nop_dropout,eliminate_nop_flatten,eliminate_nop_monoto` |
| task077 | sim | 7657 | 7657 | 0 | 15->15 | 11->11 | 979->1403 | False | `onnxsim(input=1x10x30x30)` |
| task077 | optimizer_sim | 7657 | 7657 | 0 | 15->15 | 11->11 | 979->1403 | False | `eliminate_deadend,eliminate_identity,eliminate_nop_cast,eliminate_nop_dropout,eliminate_nop_flatten,eliminate_nop_monoto` |
| task096 | baseline | 7678 | 7678 | 0 | 113->113 | 33->33 | 14209->14209 | False | `` |
| task096 | optimizer | 7678 | 7678 | 0 | 113->113 | 33->33 | 14209->13981 | False | `eliminate_deadend,eliminate_identity,eliminate_nop_cast,eliminate_nop_dropout,eliminate_nop_flatten,eliminate_nop_monoto` |
| task096 | sim | 7678 | 7678 | 0 | 113->113 | 33->33 | 14209->14043 | False | `onnxsim(input=1x10x30x30)` |
| task096 | optimizer_sim | 7678 | 7678 | 0 | 113->113 | 33->33 | 14209->14043 | False | `eliminate_deadend,eliminate_identity,eliminate_nop_cast,eliminate_nop_dropout,eliminate_nop_flatten,eliminate_nop_monoto` |
| task009 | baseline | 6694 | 6694 | 0 | 23->23 | 15->15 | 2039->2039 | False | `` |
| task009 | optimizer | 6694 | 6694 | 0 | 23->23 | 15->15 | 2039->2039 | False | `eliminate_deadend,eliminate_identity,eliminate_nop_cast,eliminate_nop_dropout,eliminate_nop_flatten,eliminate_nop_monoto` |
| task009 | sim | 6694 | 6694 | 0 | 23->23 | 15->15 | 2039->2794 | False | `onnxsim(input=1x10x30x30)` |
| task009 | optimizer_sim | 6694 | 6694 | 0 | 23->23 | 15->15 | 2039->2794 | False | `eliminate_deadend,eliminate_identity,eliminate_nop_cast,eliminate_nop_dropout,eliminate_nop_flatten,eliminate_nop_monoto` |
| task383 | baseline | 5830 | 5830 | 0 | 54->54 | 12->12 | 4120->4120 | False | `` |
| task383 | optimizer | 5830 | 5830 | 0 | 54->54 | 12->12 | 4120->4120 | False | `eliminate_deadend,eliminate_identity,eliminate_nop_cast,eliminate_nop_dropout,eliminate_nop_flatten,eliminate_nop_monoto` |
| task383 | sim | 5830 | 5830 | 0 | 54->54 | 12->12 | 4120->4120 | False | `onnxsim(input=1x10x30x30)` |
| task383 | optimizer_sim | 5830 | 5830 | 0 | 54->54 | 12->12 | 4120->4120 | False | `eliminate_deadend,eliminate_identity,eliminate_nop_cast,eliminate_nop_dropout,eliminate_nop_flatten,eliminate_nop_monoto` |
| task382 | baseline | 5695 | 5695 | 0 | 71->71 | 25->25 | 6737->6737 | False | `` |
| task382 | optimizer | 5695 | 5695 | 0 | 71->71 | 25->25 | 6737->6737 | False | `eliminate_deadend,eliminate_identity,eliminate_nop_cast,eliminate_nop_dropout,eliminate_nop_flatten,eliminate_nop_monoto` |
| task382 | sim | 5695 | 5695 | 0 | 71->71 | 25->25 | 6737->9249 | False | `onnxsim(input=1x10x30x30)` |
| task382 | optimizer_sim | 5695 | 5695 | 0 | 71->71 | 25->25 | 6737->9249 | False | `eliminate_deadend,eliminate_identity,eliminate_nop_cast,eliminate_nop_dropout,eliminate_nop_flatten,eliminate_nop_monoto` |
| task278 | baseline | 4505 | 4505 | 0 | 10->10 | 8->8 | 751->751 | False | `` |
| task278 | optimizer | 4505 | 4505 | 0 | 10->10 | 8->8 | 751->751 | False | `eliminate_deadend,eliminate_identity,eliminate_nop_cast,eliminate_nop_dropout,eliminate_nop_flatten,eliminate_nop_monoto` |
| task278 | sim | 4505 | 4505 | 0 | 10->10 | 8->8 | 751->1031 | False | `onnxsim(input=1x10x30x30)` |
| task278 | optimizer_sim | 4505 | 4505 | 0 | 10->10 | 8->8 | 751->1031 | False | `eliminate_deadend,eliminate_identity,eliminate_nop_cast,eliminate_nop_dropout,eliminate_nop_flatten,eliminate_nop_monoto` |
| task165 | baseline | 4544 | 4544 | 0 | 37->37 | 20->20 | 2323->2323 | False | `` |
| task165 | optimizer | 4544 | 4544 | 0 | 37->37 | 20->20 | 2323->2323 | False | `eliminate_deadend,eliminate_identity,eliminate_nop_cast,eliminate_nop_dropout,eliminate_nop_flatten,eliminate_nop_monoto` |
| task165 | sim | 4544 | 4544 | 0 | 37->37 | 20->20 | 2323->3564 | False | `onnxsim(input=1x10x30x30)` |
| task165 | optimizer_sim | 4544 | 4544 | 0 | 37->37 | 20->20 | 2323->3564 | False | `eliminate_deadend,eliminate_identity,eliminate_nop_cast,eliminate_nop_dropout,eliminate_nop_flatten,eliminate_nop_monoto` |
| task378 | baseline | 3091 | 3091 | 0 | 63->63 | 21->21 | 5072->5072 | False | `` |
| task378 | optimizer | 3091 | 3091 | 0 | 63->63 | 21->21 | 5072->5072 | False | `eliminate_deadend,eliminate_identity,eliminate_nop_cast,eliminate_nop_dropout,eliminate_nop_flatten,eliminate_nop_monoto` |
| task378 | sim | 3091 | 3091 | 0 | 63->63 | 21->21 | 5072->5072 | False | `onnxsim(input=1x10x30x30)` |
| task378 | optimizer_sim | 3091 | 3091 | 0 | 63->63 | 21->21 | 5072->5072 | False | `eliminate_deadend,eliminate_identity,eliminate_nop_cast,eliminate_nop_dropout,eliminate_nop_flatten,eliminate_nop_monoto` |
| task132 | baseline | 3652 | 3652 | 0 | 67->67 | 14->14 | 5073->5073 | False | `` |
| task132 | optimizer | 3652 | 3652 | 0 | 67->67 | 14->14 | 5073->5073 | False | `eliminate_deadend,eliminate_identity,eliminate_nop_cast,eliminate_nop_dropout,eliminate_nop_flatten,eliminate_nop_monoto` |
| task132 | sim | 3652 | 3652 | 0 | 67->67 | 14->14 | 5073->7061 | False | `onnxsim(input=1x10x30x30)` |
| task132 | optimizer_sim | 3652 | 3652 | 0 | 67->67 | 14->14 | 5073->7061 | False | `eliminate_deadend,eliminate_identity,eliminate_nop_cast,eliminate_nop_dropout,eliminate_nop_flatten,eliminate_nop_monoto` |
