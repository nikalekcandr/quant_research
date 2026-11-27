@echo off
rem One click: update only the code of this folder. data\ is never touched (it comes from the exchanges).
rem Windows keeps the line endings below as CRLF, so cmd.exe reads them normally.
setlocal
cd /d "%~dp0"
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
echo.
"%PY%" scripts\get_code.py %*
echo.
echo Готово. Нажмите любую клавишу, чтобы закрыть окно.
pause >nul
