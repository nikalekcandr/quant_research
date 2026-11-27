@echo off
rem ---------------------------------------------------------------------------
rem Один файл: двойной клик — и всё.
rem   1) находит и останавливает забытые закачки прошлых запусков
rem   2) докачивает все датасеты (Kalshi: свечи за весь час вместо 10 минут)
rem   3) показывает прогресс построчно в этом окне
rem   4) пересобирает таблицы и проверяет, что легло на диск
rem
rem Полезное: download_all.bat --dry-run   (только план)
rem            download_all.bat --verify    (только проверка)
rem            download_all.bat --status    (итог прошлого прогона)
rem ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0.."
title Закачка данных quant_research
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
echo.
"%PY%" scripts\download_all.py %*
echo.
echo Готово. Итог: python scripts\download_all.py --status
echo Нажмите любую клавишу, чтобы закрыть окно.
pause >nul
