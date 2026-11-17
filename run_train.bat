@echo off
chcp 65001 >nul
setlocal
set PY=E:\anaconda3\envs\pytorch\python.exe
cd /d "%~dp0"
echo [1/1] 开始训练 YOLOv11（配置见 configs\train.yaml）
"%PY%" scripts\train.py %*
pause
