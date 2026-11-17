@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM ---- locate python: activated env > classroom env > this machine's env > PATH ----
set "PY="
if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" set "PY=%CONDA_PREFIX%\python.exe"

if not defined PY for %%D in (
    "%USERPROFILE%\miniconda3"
    "%USERPROFILE%\anaconda3"
    "C:\ProgramData\miniconda3"
    "C:\ProgramData\anaconda3"
    "E:\anaconda3"
    "D:\anaconda3"
) do (
    if not defined PY if exist "%%~D\envs\classroom\python.exe" set "PY=%%~D\envs\classroom\python.exe"
)

if not defined PY if exist "E:\anaconda3\envs\pytorch\python.exe" set "PY=E:\anaconda3\envs\pytorch\python.exe"

if not defined PY where python >nul 2>nul && set "PY=python"

if not defined PY (
    echo.
    echo [ERROR] Python not found.
    echo   1^) Open Anaconda Prompt
    echo   2^) conda activate classroom
    echo   3^) python app\desktop_app.py
    echo.
    pause
    exit /b 1
)

REM ---- make sure torch is importable, otherwise the error is cryptic ----
%PY% -c "import torch" >nul 2>nul
if errorlevel 1 (
    echo.
    echo [ERROR] This python has no torch: %PY%
    echo Run first:  conda activate classroom
    echo Or install:  pip install -r requirements.lock.txt
    echo.
    pause
    exit /b 1
)

echo Python: %PY%
echo.
%PY% app\desktop_app.py %*
echo.
pause
