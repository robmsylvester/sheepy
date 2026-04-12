# Sheepy Demo — PowerShell launcher
# Run from the repo root: .\examples\run_demo.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "`n=== Installing sheepy in editable mode ===" -ForegroundColor Cyan
pip install -e ".[dev,vector,rag]"

Write-Host "`n=== Running demo ===" -ForegroundColor Cyan
python examples/demo.py
