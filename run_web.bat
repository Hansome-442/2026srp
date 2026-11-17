@echo off
chcp 65001 >nul
setlocal
set PY=E:\anaconda3\envs\pytorch\python.exe
cd /d "%~dp0"
echo [1/1] 启动课堂行为检测系统（网页端 Streamlit）
"%PY%" -m streamlit run app\web_app.py %*
pause
