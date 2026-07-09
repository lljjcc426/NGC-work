param(
    [string]$KaggleGolfRoot = "E:\kagglegolf",
    [string]$Owner = "C"
)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $Here "..\..")
$AssignmentCsv = Join-Path $RepoRoot "assignments\task_assignment_400.csv"
$Exporter = Join-Path $KaggleGolfRoot "scripts\neurogolf\07_export_owner_dashboard.py"

if (-not (Test-Path $Exporter)) {
    throw "Exporter not found: $Exporter"
}

Push-Location $KaggleGolfRoot
try {
    python $Exporter --owner $Owner --assignment-csv $AssignmentCsv --out-dir $Here
}
finally {
    Pop-Location
}
