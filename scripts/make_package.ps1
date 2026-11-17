# 工程打包脚本：把代码、配置、文档、报告和最终模型打成一个 zip
# 不含数据集（data/processed）与下载的原始数据（data/raw/downloads），体积可控、可直接分发。
#
# 用法：
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\make_package.ps1
#   powershell ... -File scripts\make_package.ps1 -WithModel:$false   # 不打包权重

param(
    [string]$OutZip = "",
    [switch]$WithModel = $true
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$stamp = Get-Date -Format "yyyyMMdd"
$stageRoot = Join-Path $env:TEMP "srp_package_$stamp"
$stage = Join-Path $stageRoot "2026srp_classroom_behavior"
if (Test-Path $stageRoot) { Remove-Item -Recurse -Force $stageRoot }
New-Item -ItemType Directory -Force -Path $stage | Out-Null

function Copy-Tree([string]$rel, [string[]]$excludeDirs = @()) {
    $src = Join-Path $Root $rel
    if (-not (Test-Path $src)) { return }
    $dst = Join-Path $stage $rel
    Copy-Item -Recurse -Force $src $dst
    if ($excludeDirs.Count -gt 0) {
        foreach ($ex in $excludeDirs) {
            Get-ChildItem $dst -Recurse -Directory -Filter $ex -ErrorAction SilentlyContinue |
                Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
    Write-Host "  + $rel"
}

Write-Host "收集代码与文档..."
foreach ($d in @("src", "scripts", "app", "configs", "docs")) {
    Copy-Tree $d @("__pycache__")
}
foreach ($f in @("README.md", "requirements.txt", ".gitignore")) {
    if (Test-Path (Join-Path $Root $f)) { Copy-Item (Join-Path $Root $f) (Join-Path $stage $f); Write-Host "  + $f" }
}
Get-ChildItem $Root -Filter *.bat -File | ForEach-Object {
    Copy-Item $_.FullName (Join-Path $stage $_.Name); Write-Host "  + $($_.Name)"
}

Write-Host "收集报告（排除大日志）..."
$repDst = Join-Path $stage "reports"
New-Item -ItemType Directory -Force -Path $repDst | Out-Null
Get-ChildItem (Join-Path $Root "reports") -File |
    Where-Object { $_.Extension -in @(".md", ".json", ".png", ".jpg", ".csv") } |
    ForEach-Object { Copy-Item $_.FullName (Join-Path $repDst $_.Name) }
Write-Host "  + reports/（$((Get-ChildItem $repDst -File).Count) 个文件）"

Write-Host "收集数据说明（不含数据本体）..."
$dataDst = Join-Path $stage "data"
New-Item -ItemType Directory -Force -Path $dataDst | Out-Null
if (Test-Path (Join-Path $Root "data\dataset.yaml")) {
    Copy-Item (Join-Path $Root "data\dataset.yaml") (Join-Path $dataDst "dataset.yaml")
    Write-Host "  + data/dataset.yaml"
}
if (Test-Path (Join-Path $Root "data\raw\README.md")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $dataDst "raw") | Out-Null
    Get-ChildItem (Join-Path $Root "data\raw") -File | ForEach-Object {
        Copy-Item $_.FullName (Join-Path $dataDst "raw\$($_.Name)")
    }
    Write-Host "  + data/raw/*.md, classes.txt 等说明文件"
}

if ($WithModel) {
    Write-Host "收集最终模型..."
    $mDst = Join-Path $stage "runs"
    New-Item -ItemType Directory -Force -Path $mDst | Out-Null
    if (Test-Path (Join-Path $Root "runs\best.pt")) {
        Copy-Item (Join-Path $Root "runs\best.pt") (Join-Path $mDst "best.pt")
        Write-Host "  + runs/best.pt"
    }
}

# 附带一份环境与版本信息
$info = @()
$info += "打包时间: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
$info += "工程路径: $Root"
try { $info += "Python: " + (& "E:\anaconda3\envs\pytorch\python.exe" --version 2>&1) } catch {}
try {
    $info += "torch/ultralytics: " + (& "E:\anaconda3\envs\pytorch\python.exe" -c "import torch,ultralytics;print(torch.__version__, ultralytics.__version__)" 2>&1)
} catch {}
$info += "GPU: " + (nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1)
$info | Set-Content (Join-Path $stage "ENVIRONMENT.txt") -Encoding utf8

if (-not $OutZip) { $OutZip = Join-Path $Root ("2026srp_classroom_behavior_$stamp.zip") }
if (Test-Path $OutZip) { Remove-Item $OutZip -Force }
Write-Host "压缩中 -> $OutZip"
Compress-Archive -Path $stage -DestinationPath $OutZip -CompressionLevel Optimal
Remove-Item -Recurse -Force $stageRoot

$size = [math]::Round((Get-Item $OutZip).Length / 1MB, 1)
Write-Host "完成：$OutZip（$size MB）"
