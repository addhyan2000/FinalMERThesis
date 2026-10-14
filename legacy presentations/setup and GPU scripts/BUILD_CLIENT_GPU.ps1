# Package MER_Client_GPU for client delivery (run from this folder).
# Creates a clean copy + MER_Client_GPU.zip in the parent directory.

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$ParentDir = Split-Path $ProjectRoot -Parent
$OutDir = Join-Path $ParentDir "MER_Client_GPU_dist"
$ZipPath = Join-Path $ParentDir "MER_Client_GPU.zip"

Write-Host "Packaging client GPU delivery..." -ForegroundColor Cyan
Write-Host "  Source: $ProjectRoot"
Write-Host "  Staging: $OutDir"
Write-Host "  Zip:     $ZipPath"

if (Test-Path $OutDir) {
    Remove-Item $OutDir -Recurse -Force
}
New-Item -ItemType Directory -Path $OutDir | Out-Null

robocopy $ProjectRoot $OutDir /E /NFL /NDL /NJH /NJS /NC /NS /NP `
    /XD .git .venv __pycache__ env build dist .vscode .idea MER_Client_GPU_dist `
    /XF *.log *.pyc *.pyo gui_settings.json `
    | Out-Null

# Empty Processed_Data (no stale checkpoints / CSV / tensors)
$ProcessedRoot = Join-Path $OutDir "Processed_Data"
if (Test-Path $ProcessedRoot) {
    Remove-Item $ProcessedRoot -Recurse -Force
}
foreach ($sub in @("checkpoints", "intermediates", "tensors", "tensors_raw", "smoke_tensors")) {
    New-Item -ItemType Directory -Path (Join-Path $ProcessedRoot $sub) -Force | Out-Null
}
".gitkeep" | Set-Content -Path (Join-Path $ProcessedRoot ".gitkeep") -Encoding UTF8

# Empty ablation results
foreach ($rel in @("Ablation_Study\results", "Ablation_Study\results_individual", "Ablation_Study\logs")) {
    $dir = Join-Path $OutDir $rel
    if (Test-Path $dir) { Remove-Item $dir -Recurse -Force }
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
    ".gitkeep" | Set-Content -Path (Join-Path $dir ".gitkeep") -Encoding UTF8
}

if (Test-Path (Join-Path $OutDir "tmp")) {
    Remove-Item (Join-Path $OutDir "tmp") -Recurse -Force
}

Get-ChildItem $OutDir -Recurse -Filter "*.log" -ErrorAction SilentlyContinue | Remove-Item -Force

if (Test-Path $ZipPath) {
    Remove-Item $ZipPath -Force
}
Compress-Archive -Path "$OutDir\*" -DestinationPath $ZipPath -Force

$zipMb = [math]::Round((Get-Item $ZipPath).Length / 1MB, 1)
Write-Host "`nDone." -ForegroundColor Green
Write-Host "  Folder: $OutDir"
Write-Host "  Zip:    $ZipPath (${zipMb} MB)"
Write-Host "`nSend MER_Client_GPU.zip to the client." -ForegroundColor Cyan
