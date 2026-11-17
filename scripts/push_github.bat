@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0.."

REM 优先用本机安装的 git；没有就用 DSH 附带的这份
set GIT=git
where git >nul 2>nul || set GIT=C:\Users\123\.local\mingit\cmd\git.exe

echo ============================================
echo  推送到 GitHub： git@github.com:Hansome-442/2026srp.git
echo ============================================
echo.

"%GIT%" status --short
echo.
echo 上面如果有改动，会先提交再推送。
set /p MSG=提交说明（直接回车用默认）：
if "%MSG%"=="" set MSG=chore: 更新
if not "%MSG%"=="skip" (
  "%GIT%" add -A
  "%GIT%" -c user.name="2026srp" -c user.email="2026srp@localhost" commit -m "%MSG%" 2>nul
)

echo.
echo 正在推送...
"%GIT%" push -u github main
echo.
if errorlevel 1 (
  echo [失败] 常见原因：
  echo   1. GitHub 上还没建这个仓库 —— 先去 https://github.com/new?name=2026srp 建一个空库
  echo   2. SSH key 没加到 GitHub —— 检查 https://github.com/settings/keys
) else (
  echo [成功] 打开 https://github.com/Hansome-442/2026srp 看看
)
pause
