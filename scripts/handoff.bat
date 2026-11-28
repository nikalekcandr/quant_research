@echo off
rem Одна команда: проверить окружение, досчитать всё несчитанное, собрать отчёт, проверить его числа.
rem Двойной клик по этому файлу делает то же самое. Результат - HANDOFF.md в корне проекта.
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONWARNINGS=once
cd /d "%~dp0.."
python scripts\handoff.py %*
if errorlevel 2 (
  echo.
  echo Есть расхождения или непройденные исследования - подробности в HANDOFF.md
) else (
  echo.
  echo Готово: HANDOFF.md
)
pause
