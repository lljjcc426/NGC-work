# Public Research Brief

Status: BLOCKED before Kaggle kernel/discussion/writeup ingest because `KAGGLE_API_TOKEN` is not set.

No public notebook title, author, vote count, score, discussion content, feature, model, or post-processing claim has been inferred.

## Required Sections

1. Public notebook list: UNKNOWN
2. Notebook author, title, link, votes, public score: UNKNOWN
3. High-value discussion information: UNKNOWN
4. Strong public baseline availability: UNKNOWN
5. Public solution features/models/post-processing: UNKNOWN
6. Three notebooks worth reproducing: UNKNOWN
7. Unconfirmed content: all content pending Kaggle access

## Commands To Run After Token Is Set

```powershell
Push-Location external\nvidia-kaggle\skills\nvidia-kaggle-skill
python .\scripts\kernel_ingest.py neurogolf-2026 --max-pages 3 --sort-by voteCount --page-size 40
python .\scripts\kernel_query.py neurogolf-2026 --limit 30 --as-json > ..\..\..\..\reports\top_kernels.json
python .\scripts\fetch_top_kernel_scores.py neurogolf-2026 --sort descending > ..\..\..\..\reports\top_kernel_scores_desc.json
python .\scripts\fetch_top_kernel_scores.py neurogolf-2026 --sort ascending > ..\..\..\..\reports\top_kernel_scores_asc.json
python .\scripts\discussion_ingest.py neurogolf-2026 --max-pages 3 --sort-by hotness --page-size 40
python .\scripts\discussion_query.py neurogolf-2026 --limit 30 --as-json > ..\..\..\..\reports\top_discussions.json
python .\scripts\fetch_leaderboard_writeups.py https://www.kaggle.com/competitions/neurogolf-2026/leaderboard > ..\..\..\..\reports\writeup_links.json
Pop-Location
```
