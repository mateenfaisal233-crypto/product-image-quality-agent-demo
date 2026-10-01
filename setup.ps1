# ============================================================
# One-shot setup (client machine) - chalana:
#   powershell -ExecutionPolicy Bypass -File setup.ps1
# Sab kuch isi folder mein install hota hai (.venv + .modelcache).
# ============================================================
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$py = "python"
if (-not (Get-Command $py -ErrorAction SilentlyContinue)) { $py = "py" }
if (-not (Get-Command $py -ErrorAction SilentlyContinue)) {
    Write-Host "Python nahi mila. Python 3.10+ install karein: https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

Write-Host "[1/4] venv bana rahe hain..." -ForegroundColor Cyan
& $py -m venv .venv
if (-not (Test-Path ".venv\Scripts\python.exe")) { Write-Host "venv fail" -ForegroundColor Red; exit 1 }
$vpy = ".venv\Scripts\python.exe"

Write-Host "[2/4] torch (CPU)..." -ForegroundColor Cyan
& $vpy -m pip install --upgrade pip --quiet
& $vpy -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

Write-Host "[3/4] baqi packages..." -ForegroundColor Cyan
& $vpy -m pip install --no-cache-dir -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Host "pip install fail" -ForegroundColor Red; exit 1 }

Write-Host "[4/4] sample images bana rahe hain..." -ForegroundColor Cyan
& $vpy examples\make_samples.py --out examples\samples

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "  Demo chalane ke liye:"
Write-Host "    .\.venv\Scripts\python.exe cli.py --image examples\samples\doliprane_ok.jpg --json examples\samples\product_doliprane.json"
Write-Host "  API server:"
Write-Host "    .\.venv\Scripts\python.exe cli.py serve --port 8000"
Write-Host "  Tests:"
Write-Host "    .\.venv\Scripts\python.exe -m pytest tests -q"
