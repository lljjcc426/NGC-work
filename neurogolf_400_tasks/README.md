# NeuroGolf 2026 Full Task Mirror

Generated at: 2026-07-09T14:31:04

Source: Kaggle competition data downloaded with `kaggle competitions download -c neurogolf-2026` into `E:/kagglegolf/data/raw/neurogolf-2026`.

This directory mirrors the 400 public task JSON files from the competition package. Each task JSON contains the full train, arc-gen, and test examples present in the downloaded Kaggle data, including input and output grids.

Open the static viewer through a local server:

```powershell
cd E:\kongming\NGC-work\neurogolf_400_tasks
python -m http.server 8770 --bind 127.0.0.1
```

Then open `http://127.0.0.1:8770/viewer.html`.

## Files

- `tasks/task001.json` ... `tasks/task400.json`: complete task JSON files.
- `task_index.csv`: file size, SHA256, example counts, and grid shape summary.
- `task_index.md`: compact index table.
- `viewer.html`: browser viewer for all 400 task JSON files.
- `task_visibility_sources.md`: evidence and commands used to confirm full task visibility.

## Sample Index

| task | bytes | train | arc-gen | test | input shapes | output shapes |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| task001 | 85526 | 5 | 262 | 1 | `3x3` | `9x9` |
| task002 | 499596 | 5 | 262 | 1 | `10x10;11x11;12x12;13x13;14x14;15x15;16x16;17x17;18x18;19x19;20x20;6x6;7x7;8x8;9x` | `10x10;11x11;12x12;13x13;14x14;15x15;16x16;17x17;18x18;19x19;20x20;6x6;7x7;8x8;9x` |
| task003 | 50384 | 3 | 261 | 1 | `6x3` | `9x3` |
| task004 | 246889 | 2 | 262 | 1 | `10x10;10x12;10x13;10x14;10x16;10x8;10x9;11x10;11x11;11x12;11x13;11x14;11x15;11x1` | `10x10;10x12;10x13;10x14;10x16;10x8;10x9;11x10;11x11;11x12;11x13;11x14;11x15;11x1` |
| task005 | 732864 | 3 | 262 | 1 | `21x21` | `21x21` |
| task006 | 33816 | 3 | 262 | 1 | `3x7` | `3x3` |
| task007 | 92336 | 3 | 262 | 1 | `7x7` | `7x7` |
| task008 | 246742 | 3 | 262 | 1 | `10x10;10x11;10x12;10x13;10x14;10x15;10x16;10x8;10x9;11x10;11x11;11x12;11x13;11x1` | `10x10;10x11;10x12;10x13;10x14;10x15;10x16;10x8;10x9;11x10;11x11;11x12;11x13;11x1` |
| task009 | 870037 | 3 | 261 | 1 | `17x17;20x20;23x23;26x26;29x29` | `17x17;20x20;23x23;26x26;29x29` |
| task010 | 144989 | 2 | 262 | 1 | `9x9` | `9x9` |
| task011 | 212299 | 4 | 262 | 1 | `11x11` | `11x11` |
| task012 | 248339 | 2 | 262 | 1 | `12x12` | `12x12` |
| task013 | 377827 | 4 | 262 | 1 | `10x20;10x21;10x22;10x23;10x24;10x25;10x26;10x27;10x29;11x20;11x21;11x22;11x24;11` | `10x20;10x21;10x22;10x23;10x24;10x25;10x26;10x27;10x29;11x20;11x21;11x22;11x24;11` |
| task014 | 386534 | 3 | 262 | 1 | `15x15;15x16;15x17;15x19;15x21;15x22;15x23;15x24;15x25;16x15;16x16;16x18;16x19;16` | `10x10;10x11;10x12;10x13;10x5;10x6;10x7;10x8;11x10;11x11;11x5;11x6;11x7;11x8;11x9` |
| task015 | 144989 | 3 | 261 | 1 | `9x9` | `9x9` |
| task016 | 24331 | 4 | 262 | 1 | `3x3` | `3x3` |
| task017 | 732864 | 3 | 262 | 1 | `21x21` | `21x21` |
| task018 | 679972 | 3 | 262 | 1 | `12x15;12x18;12x20;12x21;12x22;13x16;13x18;13x20;13x21;13x23;13x24;14x15;14x18;14` | `12x15;12x18;12x20;12x21;12x22;13x16;13x18;13x20;13x21;13x23;13x24;14x15;14x18;14` |
| task019 | 78793 | 4 | 262 | 1 | `2x2;2x3;2x4;2x5;2x6;3x2;3x3;3x4;3x5;3x6;4x2;4x3;4x4;4x5;4x6;5x2;5x3;5x4;5x5;5x6;` | `10x10;10x12;10x4;10x6;10x8;12x10;12x12;12x4;12x6;12x8;4x10;4x12;4x4;4x6;4x8;6x10` |
| task020 | 176924 | 3 | 262 | 1 | `10x10` | `10x10` |

See `task_index.csv` for all 400 rows.
