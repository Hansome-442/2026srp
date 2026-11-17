# 全自动流水线：重建数据集 -> 去重检查 -> 训练 -> 过拟合诊断 -> 测试集评估 -> TTA 评估 -> 自检 -> 关机
#
# 用法：
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_full_pipeline.ps1
#   powershell ... -File scripts\run_full_pipeline.ps1 -RunName classroom_v3_100ep -NoShutdown
#
# 数据集来源写在 configs\dataset_sources.json 里，格式：
#   [ {"dir": "data/raw/downloads/xxx", "names": "writing,reading,..."} ]

param(
    [string]$SourceSpec = "configs\dataset_sources.json",
    [string]$TrainConfig = "configs\train_v3.yaml",
    [string]$RunName = "classroom_v3_100ep",
    [string]$DatasetYaml = "data\dataset.yaml",
    [string]$OutDir = "data\processed",
    [int]$DelaySeconds = 180,
    [switch]$NoShutdown,
    [switch]$SkipPrepare,
    [double]$BaselineMap5095 = 0,
    [string]$RollbackWeights = "",
    [int]$MinMap50 = 0
)

$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$PY = "E:\anaconda3\envs\pytorch\python.exe"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Log = Join-Path $Root "reports\pipeline.log"

function Write-Log([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $Log -Value $line -Encoding utf8
}

# 防止训练期间睡眠
try { powercfg /change standby-timeout-ac 0 | Out-Null } catch {}
$script:KeepAwakeOk = $false
try {
    Add-Type -Namespace Win32Power2 -Name Api2 -MemberDefinition @'
[DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint esFlags);
'@ -ErrorAction Stop
    $script:KeepAwakeOk = $true
} catch { $script:KeepAwakeOk = $false }
function Set-KeepAwake {
    if ($script:KeepAwakeOk) { [Win32Power2.Api2]::SetThreadExecutionState(0x80000001) | Out-Null }
}

Write-Log "===== pipeline start: run=$RunName config=$TrainConfig ====="
Set-KeepAwake

# ---------------- 1) 重建数据集 ----------------
$specPath = Join-Path $Root $SourceSpec
if (-not (Test-Path $specPath)) { Write-Log "ERROR: 找不到数据源配置 $specPath"; exit 2 }
$sources = Get-Content $specPath -Raw -Encoding utf8 | ConvertFrom-Json

$argsList = @()
foreach ($s in $sources) {
    $dir = $s.dir
    if (-not (Test-Path (Join-Path $Root $dir))) { Write-Log "WARN: 数据源不存在，跳过 $dir"; continue }
    $argsList += "--yolo-dir"; $argsList += $dir
    if ($s.names) { $argsList += "--yolo-names"; $argsList += $s.names }
}
if ($argsList.Count -eq 0) { Write-Log "ERROR: 没有任何可用数据源"; exit 2 }

if ($SkipPrepare) { Write-Log "SkipPrepare：沿用现有 data\processed 与 dataset.yaml，不重建数据集" } else {
Write-Log "重建数据集（去重分组 + 来源/类别双层分层）..."
if (Test-Path (Join-Path $Root $OutDir)) { Remove-Item -Recurse -Force (Join-Path $Root $OutDir) }
& $PY "scripts\prepare_dataset.py" @argsList --out $OutDir --dataset-yaml $DatasetYaml *> "reports\pipeline_prepare.log"
$rc = $LASTEXITCODE
Write-Log "prepare_dataset exit=$rc（日志 reports\pipeline_prepare.log）"
if ($rc -ne 0) { Write-Log "ERROR: 数据集重建失败，终止（不关机）"; exit 3 }
Set-KeepAwake

# ---------------- 2) 去重/泄漏检查 ----------------
& $PY "scripts\dataset_dedup.py" --data $DatasetYaml *> "reports\pipeline_dedup.log"
Write-Log "dedup check exit=$LASTEXITCODE（报告 reports\dataset_dedup.md）"
Set-KeepAwake
}   # end if (-not $SkipPrepare)

# ---------------- 3) 训练 ----------------
Write-Log "开始训练：config=$TrainConfig name=$RunName"
$trainStart = Get-Date
& $PY "scripts\train.py" --config $TrainConfig --name $RunName *> "reports\pipeline_train.log"
$trainRc = $LASTEXITCODE
$dur = ((Get-Date) - $trainStart).TotalHours
Write-Log ("训练结束 exit={0}，用时 {1:N2} 小时（日志 reports\pipeline_train.log）" -f $trainRc, $dur)
Set-KeepAwake

$runDir = Join-Path $Root ("runs\train\" + $RunName)
$best = Join-Path $runDir "weights\best.pt"
if (-not (Test-Path $best)) {
    Write-Log "ERROR: 没找到 best.pt，训练可能失败；不关机，便于排查"
    exit 4
}

# ---------------- 4) 过拟合诊断 ----------------
& $PY "scripts\check_overfit.py" --run ("runs\train\" + $RunName) --json ("reports\overfit_" + $RunName + ".json") *> "reports\pipeline_overfit.log"
Write-Log "过拟合诊断 exit=$LASTEXITCODE（报告 reports\overfit_$RunName.md）"
Set-KeepAwake

# ---------------- 5) 测试集评估（普通 + TTA） ----------------
& $PY "scripts\val.py" --weights "runs\best.pt" --data $DatasetYaml --split test --imgsz 640 --batch 8 --name ("final_" + $RunName) *> "reports\pipeline_val.log"
Write-Log "测试集评估 exit=$LASTEXITCODE（报告 reports\val_report_test.md）"
Set-KeepAwake

& $PY "scripts\val.py" --weights "runs\best.pt" --data $DatasetYaml --split test --imgsz 640 --batch 8 --name ("tta_" + $RunName) --tta *> "reports\pipeline_val_tta.log"
Write-Log "TTA 评估 exit=$LASTEXITCODE（日志 reports\pipeline_val_tta.log）"
# ---- 与基线比较：新模型更差则回滚 ----
try {
    $reportPath = Join-Path $Root "reports\val_report_test.md"
    if (Test-Path $reportPath) {
        $rep = Get-Content $reportPath -Raw -Encoding utf8
        if ($rep -match 'mAP50-95：\*\*([0-9.]+)\*\*') {
            $newMap = [double]$Matches[1]
            Write-Log "本次测试集 mAP50-95 = $newMap（基线 $BaselineMap5095）"
            if ($BaselineMap5095 -gt 0 -and $newMap -lt $BaselineMap5095 -and $RollbackWeights -and (Test-Path (Join-Path $Root $RollbackWeights))) {
                Copy-Item (Join-Path $Root $RollbackWeights) (Join-Path $Root "runs\best.pt") -Force
                Write-Log "WARNING: 新模型不如基线，已回滚 runs\best.pt -> $RollbackWeights（部署用旧模型）"
            } elseif ($BaselineMap5095 -gt 0) {
                Write-Log "新模型优于基线，保留新权重"
            }
        }
    }
} catch { Write-Log "基线比较失败：$($_.Exception.Message)" }

Set-KeepAwake

# ---------------- 6) 端到端自检 ----------------
& $PY "scripts\smoke_test.py" --weights "runs\best.pt" --source "data\processed\images\val" *> "reports\pipeline_smoke.log"
Write-Log "自检 exit=$LASTEXITCODE（日志 reports\pipeline_smoke.log）"

# ---------------- 7) 关机 ----------------
Write-Log "===== pipeline 完成 ====="
if ($NoShutdown) {
    Write-Log "NoShutdown：保持开机"
} else {
    Write-Log "全部完成，$DelaySeconds 秒后自动关机（取消：shutdown /a）"
    shutdown /s /t $DelaySeconds /c "Classroom behavior pipeline finished. Auto shutdown in $DelaySeconds s. Cancel: shutdown /a"
}
