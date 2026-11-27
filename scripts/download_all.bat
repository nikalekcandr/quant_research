@echo off
rem One click: download every dataset that is behind, then rebuild the tables.
setlocal
cd /d "%~dp0.."
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
echo.
"%PY%" scripts\download_all.py %*
echo.
echo Итог: python scripts\download_all.py --status
echo Нажмите любую клавишу, чтобы закрыть окно.
pause >nul
