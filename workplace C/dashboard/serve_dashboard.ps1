param(
    [int]$Port = 8766
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Write-Host "Serving Workplace C dashboard at http://127.0.0.1:$Port/index.html"
python -m http.server $Port --bind 127.0.0.1 --directory $Root
