$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$python = $null
foreach ($candidate in @(
    (Get-Command python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue),
    (Get-Command py -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue),
    "C:\Users\mardg\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
)) {
    if ($candidate -and (Test-Path $candidate)) { $python = $candidate; break }
}

if (-not $python) {
    throw "Python 3 was not found. Install Python 3.11+ and the packages in requirements.txt."
}

Write-Host "Starting Vertiv Knowledge at http://127.0.0.1:8000" -ForegroundColor Green
& $python app.py
