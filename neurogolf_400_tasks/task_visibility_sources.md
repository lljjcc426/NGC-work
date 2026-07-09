# Task Visibility Sources

This note records how we confirmed that all 400 NeuroGolf tasks are visible.

## Direct Kaggle data evidence

Command:

```powershell
kaggle competitions files -c neurogolf-2026 --page-size 200
```

The Kaggle CLI file listing shows individual competition files beginning with `task001.json`, `task002.json`, etc. The first page includes task files directly rather than hidden behind a generated test server.

Local download check:

```powershell
kaggle competitions download -c neurogolf-2026 -p E:\kagglegolf\data\raw
```

Downloaded archive checked at `E:/kagglegolf/data/raw/neurogolf-2026.zip`:

- zip entries: 401
- task JSON files: 400
- first task files: `task001.json` ... `task005.json`
- last task files: `task396.json` ... `task400.json`
- sample task keys: `train`, `test`, `arc-gen`
- sample `test` entries include both `input` and `output`

## Discussion / platform evidence

Kaggle discussion topics checked with `kaggle competitions topics show`:

- https://www.kaggle.com/competitions/neurogolf-2026/discussion/699313 - `Web GUI to Build ONNX by Hand`.
- https://www.kaggle.com/competitions/neurogolf-2026/discussion/699429 - `Web GUI for Hand Solving Tasks Open Source`.

These discussion threads describe browser tooling for viewing tasks and building ONNX solutions. The repository copy here uses only the downloaded Kaggle task JSON files as source of truth.

Kaggle code search checked with:

```powershell
kaggle kernels list --competition neurogolf-2026 --sort-by voteCount --page-size 100
```

Relevant high-vote public notebooks returned by the Kaggle CLI include:

- `karnakbaevarthur/all-task-description-analysis` - `All Task Description Analysis`.
- `kojimar/neurogolf-task-level-onnx-baseline` - `NeuroGolf Task-Level ONNX Baseline`.
- `franksunp/7113-80-lb-compact-onnx-artifact-view` - compact ONNX artifact viewing.

These platform results are consistent with task-level inspection being part of the public workflow, but the complete task content in this repository still comes from the downloaded Kaggle competition data files.

## Conclusion

For this competition, the local competition data package contains the full 400 task JSONs. The task JSONs are viewable directly in this repository and through `viewer.html` when served over a local HTTP server.
