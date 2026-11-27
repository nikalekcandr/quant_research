@echo off
rem One click for the night: check the code, download what is missing, write a report in the morning.
setlocal
cd /d "%~dp0.."
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
echo.
"%PY%" scripts\night.py %*
echo.
echo Утром: python scripts\night.py --status
echo Нажмите любую клавишу, чтобы закрыть окно.
pause >nul
