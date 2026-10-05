# ============================================================
# One-shot setup - runs automatically from "Start App.bat".
# Manual run:  powershell -ExecutionPolicy Bypass -File setup.ps1
# Everything installs into this folder (.venv + .modelcache).
# ============================================================
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$py = "python"
if (-not (Get-Command $py -ErrorAction SilentlyContinue)) { $py = "py" }
if (-not (Get-Command $py -ErrorAction SilentlyContinue)) {
    Write-Host "Python was not found." -ForegroundColor Red
    Write-Host "Install Python 3.10+ from https://www.python.org/downloads/"
    Write-Host "and tick 'Add Python to PATH' during installation, then run this again."
    exit 1
}

Write-Host "[1/4] Creating a private Python environment..." -ForegroundColor Cyan
& $py -m venv .venv
if (-not (Test-Path ".venv\Scripts\python.exe")) { Write-Host "Creating the environment failed." -ForegroundColor Red; exit 1 }
$vpy = ".venv\Scripts\python.exe"

Write-Host "[2/4] Installing PyTorch (CPU version, no GPU needed)..." -ForegroundColor Cyan
& $vpy -m pip install --upgrade pip --quiet
& $vpy -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

Write-Host "[3/4] Installing remaining packages..." -ForegroundColor Cyan
& $vpy -m pip install --no-cache-dir -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Host "Installing packages failed - check your internet connection." -ForegroundColor Red; exit 1 }

Write-Host "[4/4] Preparing sample images..." -ForegroundColor Cyan
& $vpy examples\make_samples.py --out examples\samples

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host 'Next step: double-click "Start App.bat" to open the app in your browser.'
