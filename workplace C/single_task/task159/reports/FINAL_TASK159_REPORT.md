# FINAL TASK159 REPORT

- Rule model: locate motif, crop source patch, tile inside the border rectangle
- Alternative structure: axis occupancy `Einsum`, `Sign`, and `ArgMax`
- Validation: `265/265`
- Parent cost: `1568`
- Candidate cost: `2000`
- Delta cost: `+432`
- Status: independently modeled; candidate rejected

The existing logarithmic scalar coordinate decoder is retained because it is
substantially cheaper than explicit row and column occupancy tensors.
