# task054 independent modeling

The task projects line frontiers from a small template into marked rectangles
and then overlays the local template neighborhood. Two direct scalar-scatter
fusion models were built to eliminate the full-grid OR mask.

They reduce nominal cost from 25394 to 23954, but pass 0/266 and 10/266. The
failure proves that duplicate-index OR semantics and marker exclusion cannot
be replaced by direct scalar overwrite or `reduction=max`.
