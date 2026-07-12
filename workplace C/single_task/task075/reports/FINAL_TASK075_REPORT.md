# FINAL TASK075 REPORT

- Rule model: template copy controlled by a 3x3 marker lattice
- Alternative structure: spatial `Slice` plus channel `ArgMax`
- Validation: `265/265`
- Parent cost: `1487`
- Candidate cost: `1788`
- Delta cost: `+301`
- Status: independently modeled; candidate rejected

The candidate proves the direct decoding rule but materializes a 90-cell
one-hot patch. The existing scalar `Einsum` is retained.
