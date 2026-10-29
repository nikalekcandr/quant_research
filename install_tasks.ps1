# Registers two Windows scheduled tasks for this project (no admin rights needed):
#   btcvol-live    every 5 minutes: order-book snapshot + live forecasts that are due (live_once.py)
#   btcvol-daily   every night at 05:10: download new data, rebuild, score live forecasts, OOS report
# Both may wake the computer from sleep. Run from the project folder:
#   powershell -ExecutionPolicy Bypass -File install_tasks.ps1
# Remove with uninstall_tasks.ps1. Logs: logs\live_YYYYMMDD.log, logs\daily_update_YYYYMMDD.log

$ErrorActionPreference = "Stop"
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path

# python: the project's virtual environment if there is one, else the python on PATH
$venv = Join-Path $dir ".venv\Scripts\pythonw.exe"
if (Test-Path $venv) {
    $pyw = $venv
    $py = Join-Path $dir ".venv\Scripts\python.exe"
} else {
    $py = (Get-Command python -ErrorAction Stop).Source
    $pyw = Join-Path (Split-Path -Parent $py) "pythonw.exe"
    if (-not (Test-Path $pyw)) { $pyw = $py }
}
Write-Host "python:  $py"
Write-Host "pythonw: $pyw (no console window every 5 minutes)"
Write-Host "project: $dir"

# quick check that the code imports before scheduling it
& $py -c "import config, live_once, daily_update, oos_report, alpha_live, alpha_extra, fast_logger; print('imports ok')"
if ($LASTEXITCODE -ne 0) { throw "python cannot import the project: run 'pip install -r requirements.txt' first" }

$settings = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 9)

# every 5 minutes, aligned to :00, :05, :10 ... (hourly forecasts are due at HH:05)
$start = (Get-Date).Date   # today 00:00
try {
    $t1 = New-ScheduledTaskTrigger -Once -At $start -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration (New-TimeSpan -Days 3650)
} catch {
    $t1 = New-ScheduledTaskTrigger -Once -At $start -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration (New-TimeSpan -Days 365)
}
$a1 = New-ScheduledTaskAction -Execute $pyw -Argument "live_once.py" -WorkingDirectory $dir
Register-ScheduledTask -TaskName "btcvol-live" -Action $a1 -Trigger $t1 -Settings $settings -Force `
    -Description "BTC volatility project: order-book snapshots and live forecasts every 5 minutes" | Out-Null

$settings2 = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 3)
$t2 = New-ScheduledTaskTrigger -Daily -At "05:10"
$a2 = New-ScheduledTaskAction -Execute $pyw -Argument "daily_update.py" -WorkingDirectory $dir
Register-ScheduledTask -TaskName "btcvol-daily" -Action $a2 -Trigger $t2 -Settings $settings2 -Force `
    -Description "BTC volatility project: nightly data update, rebuild and out-of-sample report" | Out-Null

# fast logger around 12:00 New York (Polymarket and Kalshi settle together): two triggers cover both
# 19:00 and 20:00 Moscow time (US summer / winter time); a run that is not within 75 minutes of noon exits
$settings3 = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 90)
$t3a = New-ScheduledTaskTrigger -Daily -At "17:55"
$t3b = New-ScheduledTaskTrigger -Daily -At "18:55"
$a3 = New-ScheduledTaskAction -Execute $pyw -Argument "fast_logger.py --noon-window --every 10" -WorkingDirectory $dir
Register-ScheduledTask -TaskName "btcvol-fast" -Action $a3 -Trigger @($t3a, $t3b) -Settings $settings3 -Force `
    -Description "BTC volatility project: 10-second quotes in the last hour before 12:00 New York" | Out-Null

Write-Host ""
Write-Host "Registered: btcvol-live (every 5 min), btcvol-daily (05:10), btcvol-fast (17:55 and 18:55)."
Write-Host "Check in 10 minutes:  python status.py"
Write-Host "Wake timers must be allowed: Control Panel > Power Options > Change plan settings >"
Write-Host "  Change advanced power settings > Sleep > Allow wake timers > Enable."
