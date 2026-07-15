param(
    [Parameter(Mandatory = $true)]
    [string]$HostName,

    [Parameter(Mandatory = $true)]
    [string]$User,

    [string]$RemotePath = "/opt/vertiv-knowledge",
    [int]$Port = 22,
    [string]$KeyPath = "",
    [switch]$IncludeEnv
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$archive = Join-Path $env:TEMP ("vertiv-knowledge-" + (Get-Date -Format "yyyyMMdd-HHmmss") + ".tar.gz")

$sshTarget = "$User@$HostName"
$sshArgs = @()
$scpArgs = @()
if ($Port -ne 22) {
    $sshArgs += @("-p", "$Port")
    $scpArgs += @("-P", "$Port")
}
if ($KeyPath) {
    $sshArgs += @("-i", $KeyPath)
    $scpArgs += @("-i", $KeyPath)
}

$include = @(
    "app.py",
    "indexer.py",
    "rebuild_chroma.py",
    "requirements.txt",
    "README.md",
    "static",
    "data",
    "Vertiv",
    "deploy/server.env.example",
    "deploy/vertiv-knowledge.service",
    "deploy/nginx-vertiv-knowledge.conf"
)

if ($IncludeEnv) {
    $include += ".env"
    Write-Warning "Including .env in deployment archive. This will copy secrets to the server."
}

Push-Location $root
try {
    if (Test-Path $archive) {
        Remove-Item -LiteralPath $archive -Force
    }
    tar -czf $archive --exclude="__pycache__" --exclude=".git" --exclude="*.pyc" @include
}
finally {
    Pop-Location
}

Write-Host "Uploading $archive to $sshTarget..." -ForegroundColor Cyan
scp @scpArgs $archive "${sshTarget}:/tmp/vertiv-knowledge.tar.gz"

$remote = @"
set -euo pipefail
sudo mkdir -p "$RemotePath"
sudo tar -xzf /tmp/vertiv-knowledge.tar.gz -C "$RemotePath"
sudo chown -R `$(id -u):`$(id -g) "$RemotePath"
cd "$RemotePath"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
if [ ! -f .env ]; then
  cp deploy/server.env.example .env
  echo "Created $RemotePath/.env from deploy/server.env.example; add GROQ_API_KEY before starting."
fi
echo "Deployment files copied to $RemotePath"
echo "To run manually: cd $RemotePath && VERTIV_AUTO_INDEX=0 .venv/bin/python app.py"
"@

ssh @sshArgs $sshTarget $remote

Write-Host "Done. If .env has GROQ_API_KEY, start the app manually or install the systemd service template." -ForegroundColor Green
