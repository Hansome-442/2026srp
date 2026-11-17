@echo off
chcp 65001 >nul
setlocal
set PY=E:\anaconda3\envs\pytorch\python.exe
cd /d "%~dp0"
echo [1/1] 数据转换与划分：VOC/YOLO -> data/processed
"%PY%" scripts\prepare_dataset.py %*
pause
