# FINAL TASK347 REPORT

- Rule model: union of two aligned 3x3 color masks
- Alternative structure: `Slice`, `Max`, `Sub`, `Concat`, sparse unpool
- Validation: `269/269`
- Parent cost: `143`
- Candidate cost: `275`
- Delta cost: `+132`
- Status: independently modeled; candidate rejected

The explicit union is correct but costs more intermediate memory than the
parent's boolean-template construction.
