@echo off
chcp 65001 >nul
setlocal
set PY=E:\anaconda3\envs\pytorch\python.exe
cd /d "%~dp0"

echo ============================================================
echo  课堂行为数据集转换（8 类）
echo  来源1：data\raw\classroom_voc        （7 类既有数据）
echo  来源2：data\raw\self_collected_玩手机（第 8 类自采数据）
echo ============================================================
echo.

"%PY%" scripts\prepare_dataset.py --voc-dir data\raw\classroom_voc --yolo-dir data\raw\self_collected_玩手机 --oversample-min 300

if errorlevel 1 (
  echo.
  echo [提示] 转换没有成功。最常见的原因是数据还没放进目录：
  echo        1. data\raw\classroom_voc\Annotations + JPEGImages   ^(7 类数据^)
  echo        2. data\raw\self_collected_玩手机\images + labels     ^(第 8 类数据^)
  echo        请先阅读 data\raw\00_数据放置说明.md
  echo.
)
pause
