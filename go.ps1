# Everything in one command. Safe to run again: finished steps are skipped, downloads resume where they stopped.
#   powershell -ExecutionPolicy Bypass -File go.ps1
# Log: logs\go.log. Takes the whole night (most of it is Kalshi). The computer must stay on and online.

$ErrorActionPreference = "Continue"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)
New-Item -ItemType Directory -Force logs, data | Out-Null
Start-Transcript -Append logs\go.log | Out-Null

function Step($name, [scriptblock]$body) {
    $mark = "data\.go_$name.done"
    if (Test-Path $mark) { Write-Host "== $name : already done"; return }
    Write-Host ""
    Write-Host "== $name  ($(Get-Date -Format 'dd.MM HH:mm'))"
    $global:LASTEXITCODE = 0
    & $body
    if ($LASTEXITCODE -eq 0) { New-Item -ItemType File -Force $mark | Out-Null }
    else { Write-Host "!! $name failed (exit $LASTEXITCODE): run go.ps1 again to retry this step" }
}

Step install           { python -m pip install -q -r requirements.txt }
Step tests             { python tests\test_v6.py }
Step old_logger_data   { if (Test-Path ..\data\live) { robocopy ..\data\live data\live_logger_root /E /NFL /NDL /NJH /NJS | Out-Null }; $global:LASTEXITCODE = 0 }
Step venues            { python discover_venues.py }
Step probe             { python backfill.py --probe }
Step scheduled_tasks   { powershell -ExecutionPolicy Bypass -File install_tasks.ps1 }
Step history           { python backfill.py --kalshi-from 2026-01-01 }
Step kalshi_full_hour  { python -m src.download.kalshi 2026-07-01 --minutes 60 }
Step kalshi_ranges     { python -m src.download.kalshi 2026-07-01 --series KXBTC --minutes 60 }
Step final_build       { python daily_update.py --no-download }

Write-Host ""
python status.py
Write-Host ""
Write-Host "DONE. Reports: reports\oos_report.md, reports\alpha_live.md, reports\alpha_extra.md"
Stop-Transcript | Out-Null
