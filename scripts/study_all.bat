@echo off
rem ---------------------------------------------------------------------------
rem Один файл: двойной клик — и все исследования, какие позволяют данные.
rem   1) пересобирает панель крыльев, если кэш старше свежих свечей
rem   2) запускает каждое исследование отдельным процессом (падение одного не мешает остальным)
rem   3) печатает прогресс построчно и пишет итог в logs\studies-<дата>\summary.md
rem
rem Полезное: study_all.bat --dry-run    (что будет запущено и сколько минут)
rem            study_all.bat --no-heavy   (без ML, ёмкости и опционных прогонов)
rem            study_all.bat --status     (итог прошлого запуска)
rem ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0.."
title Исследования quant_research
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONWARNINGS=once
echo.
"%PY%" scripts\study_all.py %*
echo.
echo Итог: scripts\study_all.py --status
echo Нажмите любую клавишу, чтобы закрыть окно.
pause >nul
