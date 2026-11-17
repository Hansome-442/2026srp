# 训练守护脚本：训练结束后自动做评估 + 自检，然后关机
#
# 用法（在 E:\2026srp 下执行）：
#   powershell -ExecutionPolicy Bypass -File scripts\watch_after_train.ps1 -RunName classroom_v2_100ep
#   powershell -ExecutionPolicy Bypass -File scripts\watch_after_train.ps1 -RunName classroom_v2_100ep -NoShutdown
#
# 行为：
#   1) 先关闭睡眠/休眠（训练期间机器不能被睡）；
#   2) 每 60 秒检查一次 train.py 进程与 results.csv 轮次；
#   3) 训练进程消失后：
#        - 若完成轮数 >= MinEpochs：跑 test 评估 + 端到端自检，然后 delay 秒后关机；
#        - 若不足 MinEpochs（训练崩了/被中断）：**不关机**，只写日志，方便排查。
#   4) 想临时取消关机：shutdown /a

param(
    [string]$RunName = "classroom_v2_100ep",
    [string]$Project = "runs\train",
    [int]$MinEpochs = 95,
    [int]$DelaySeconds = 180,
    [switch]$NoShutdown,
    [switch]$DryRun,
    [switch]$AutoResume,
    [int]$MaxResume = 5
)

$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$PY = "E:\anaconda3\envs\pytorch\python.exe"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$RunDir = Join-Path (Join-Path $Project $RunName) ""
$ResultsCsv = Join-Path $RunDir "results.csv"
$Log = Join-Path $Root "reports\watch_after_train.log"

function Write-Log([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $Log -Value $line -Encoding utf8
}

function Get-Epochs {
    if (Test-Path $ResultsCsv) {
        $n = (Get-Content $ResultsCsv | Measure-Object -Line).Lines - 1
        if ($n -lt 0) { return 0 }
        return $n
    }
    return -1
}

function Get-TrainingProcCount {
    $p = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*train.py*" }
    return @($p).Count
}

# ---------------- 1) 禁止睡眠/休眠 ----------------
# 先试 powercfg（部分机器需要管理员，失败也不影响后面的兜底方案）
try { powercfg /change standby-timeout-ac 0 2>$null | Out-Null } catch {}
try { powercfg /change hibernate-timeout-ac 0 2>$null | Out-Null } catch {}

# 兜底：SetThreadExecutionState（普通权限即可），告诉系统"我正在工作，别睡"
$script:KeepAwakeOk = $false
try {
    Add-Type -Namespace Win32Power -Name Api -MemberDefinition @'
[DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint esFlags);
'@ -ErrorAction Stop
    $script:KeepAwakeOk = $true
} catch {
    $script:KeepAwakeOk = $false
}
function Set-KeepAwake {
    if ($script:KeepAwakeOk) {
        # ES_CONTINUOUS(0x80000000) | ES_SYSTEM_REQUIRED(0x00000001)
        [Win32Power.Api]::SetThreadExecutionState(0x80000001) | Out-Null
    }
}
Set-KeepAwake

Write-Log "watchdog started: run=$RunName project=$Project minEpochs=$MinEpochs noShutdown=$NoShutdown"
Write-Log ("keep-awake: powercfg attempted, SetThreadExecutionState available=$($script:KeepAwakeOk)")

$lastEpoch = -1
$stallCount = 0
$script:resumeCount = 0

# ---------------- 2) 等待训练结束 ----------------
while ($true) {
    Set-KeepAwake
    $epochs = Get-Epochs
    $procs = Get-TrainingProcCount

    if ($epochs -ne $lastEpoch) {
        Write-Log "progress: epoch $epochs / 100 (train.py procs=$procs)"
        $lastEpoch = $epochs
        $stallCount = 0
    } else {
        $stallCount++
        # 进程还在、轮次 30 分钟不动 -> 记录告警（不干预）
        if ($stallCount -eq 30 -and $procs -gt 0) {
            Write-Log "WARNING: no new epoch for ~30 min, training may be stalled"
        }
    }

    if ($procs -eq 0) {
        # 训练进程消失：如果还没跑够轮次且开了 AutoResume，就自动断点续训
        if ($AutoResume -and $epochs -lt $MinEpochs -and $script:resumeCount -lt $MaxResume) {
            $script:resumeCount++
            Write-Log "training process disappeared at epoch $epochs / 100 - auto resume #$($script:resumeCount)"
            Start-Process -FilePath $PY -ArgumentList @("scripts\train.py", "--resume", "--project", $Project, "--name", $RunName) -WorkingDirectory $Root -WindowStyle Hidden
            Start-Sleep -Seconds 90
            continue
        }
        Write-Log "training process exited; final epoch = $epochs"
        break
    }
    Start-Sleep -Seconds 60
}

# ---------------- 3) 训练后的收尾 ----------------
Start-Sleep -Seconds 10
$finalEpochs = Get-Epochs
Write-Log "post-training: final epochs = $finalEpochs"

if (Test-Path (Join-Path $RunDir "weights\best.pt")) {
    if ($DryRun) {
        Write-Log "DryRun: would run test evaluation + smoke test"
    } else {
        Write-Log "best.pt found, running test evaluation + smoke test"
        & $PY scripts\val.py --weights "runs\best.pt" --data "data\dataset.yaml" --split test --imgsz 640 --batch 8 --name final_test_v2 *> "reports\val_after_train.log"
        Write-Log "val exit code = $LASTEXITCODE (log: reports\val_after_train.log)"
        & $PY scripts\smoke_test.py --weights "runs\best.pt" --source "data\processed\images\val" *> "reports\smoke_after_train.log"
        Write-Log "smoke test exit code = $LASTEXITCODE (log: reports\smoke_after_train.log)"
    }
} else {
    Write-Log "WARNING: best.pt not found - training probably failed"
}

# ---------------- 4) 关机 ----------------
if ($NoShutdown -or $DryRun) {
    Write-Log "NoShutdown/DryRun specified - keeping the machine on"
} elseif ($finalEpochs -ge $MinEpochs) {
    if ($DelaySeconds -gt 0) {
        Write-Log "SUCCESS: $finalEpochs epochs done; shutting down in $DelaySeconds seconds (cancel: shutdown /a)"
        shutdown /s /t $DelaySeconds /c "Classroom training finished ($finalEpochs/100 epochs). Auto shutdown in $DelaySeconds s. Cancel: shutdown /a"
    } else {
        Write-Log "SUCCESS: $finalEpochs epochs done; shutting down now"
        shutdown /s /t 0
    }
} else {
    Write-Log "WARNING: only $finalEpochs epochs (< $MinEpochs) - NOT shutting down. Check runs\train\$RunName and resume with:"
    Write-Log "  $PY scripts\train.py --resume --project $Project --name $RunName"
}
Write-Log "watchdog finished"
