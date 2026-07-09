param(
    [string]$Competition = "neurogolf-2026"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$SkillDir = Join-Path $ProjectRoot "external\nvidia-kaggle\skills\nvidia-kaggle-skill"

Set-Location $ProjectRoot

if (-not $env:KAGGLE_API_TOKEN) {
    throw "KAGGLE_API_TOKEN is not set. Set it before running Kaggle API/CLI workflows."
}

Write-Output "[stage 3] Fetching competition overview and dataset description"
Push-Location $SkillDir
python .\scripts\fetch_competition_info.py $Competition | Tee-Object -FilePath (Join-Path $ProjectRoot "reports\neurogolf-2026_competition_overview_raw.txt")
python .\scripts\fetch_dataset_info.py $Competition | Tee-Object -FilePath (Join-Path $ProjectRoot "reports\neurogolf-2026_dataset_description_raw.txt")
Pop-Location

Write-Output "[stage 4] Listing and downloading data"
kaggle competitions files -c $Competition | Tee-Object -FilePath "logs\kaggle_files.txt"
kaggle competitions download -c $Competition -p "data\raw"

Write-Output "[stage 4] Extracting archives"
python -c "from pathlib import Path; import zipfile; raw=Path('data/raw'); [zipfile.ZipFile(z).extractall(raw/z.stem) for z in raw.glob('*.zip')]; print('extraction complete')"

Get-ChildItem data\raw -Recurse -File -Force |
    Where-Object { $_.Name -ne '.gitkeep' } |
    Sort-Object FullName |
    ForEach-Object { $_.FullName } |
    Tee-Object -FilePath reports\data_file_manifest.txt

Write-Output "[stage 5] Profiling data"
python src\data_check.py

Write-Output "[stage 6] Researching public kernels and discussions"
Push-Location $SkillDir
python .\scripts\kernel_ingest.py $Competition --max-pages 3 --sort-by voteCount --page-size 40
python .\scripts\kernel_query.py $Competition --limit 30 --as-json | Out-File -Encoding utf8 (Join-Path $ProjectRoot "reports\top_kernels.json")
python .\scripts\fetch_top_kernel_scores.py $Competition --sort descending | Out-File -Encoding utf8 (Join-Path $ProjectRoot "reports\top_kernel_scores_desc.json")
python .\scripts\fetch_top_kernel_scores.py $Competition --sort ascending | Out-File -Encoding utf8 (Join-Path $ProjectRoot "reports\top_kernel_scores_asc.json")
python .\scripts\discussion_ingest.py $Competition --max-pages 3 --sort-by hotness --page-size 40
python .\scripts\discussion_query.py $Competition --limit 30 --as-json | Out-File -Encoding utf8 (Join-Path $ProjectRoot "reports\top_discussions.json")
python .\scripts\fetch_leaderboard_writeups.py "https://www.kaggle.com/competitions/$Competition/leaderboard" | Out-File -Encoding utf8 (Join-Path $ProjectRoot "reports\writeup_links.json")
Pop-Location

Write-Output "[stage 9] Running baseline"
python src\make_baseline.py 2>&1 | Tee-Object -FilePath logs\baseline_run.log

Write-Output "[stage 10] Validating baseline submission"
python src\validate_submission.py submissions\submission_baseline.csv

Write-Output "[stage 12] Building Kaggle kernel scaffold"
python scripts\build_kaggle_kernel.py --output-file submission.csv

Write-Output "[stage 13] Checking submission quota"
Push-Location $SkillDir
python .\scripts\submission_quota.py $Competition --by-user --by-day --as-json | Tee-Object -FilePath (Join-Path $ProjectRoot "reports\submission_quota.json")
Pop-Location

Write-Output "Pipeline complete through quota check. Do not submit until the user explicitly confirms."
