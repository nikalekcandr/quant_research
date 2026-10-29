# Registers the 24/7 alpha lab as a Windows scheduled task (no admin rights needed):
#   btcvol-lab   starts every 5 minutes (also after a reboot or sleep); a running copy is never duplicated (IgnoreNew + heartbeat),
#                so if the lab crashes or the computer wakes up, it is back within 5 minutes.
# Run from the project folder:  powershell -ExecutionPolicy Bypass -File install_lab.ps1
$ErrorActionPreference = "Stop"
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $dir
$venv = Join-Path $dir ".venv\Scripts\pythonw.exe"
if (Test-Path $venv) { $pyw = $venv; $py = Join-Path $dir ".venv\Scripts\python.exe" }
else {
    $py = (Get-Command python -ErrorAction Stop).Source
    $pyw = Join-Path (Split-Path -Parent $py) "pythonw.exe"
    if (-not (Test-Path $pyw)) { $pyw = $py }
}
& $py -m pip install -q arch scikit-learn
& $py -c "import lab.engine, lab.report, lab_daemon; print('lab imports ok')"
if ($LASTEXITCODE -ne 0) { throw "the lab does not import: send logs to Claude" }
if (-not (Test-Path (Join-Path $dir "data\lab\heartbeat.json"))) { & $py lab_daemon.py --once }
$settings = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Seconds 0) `
    -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1)
$start = (Get-Date).AddMinutes(1)
try {
    $t1 = New-ScheduledTaskTrigger -Once -At $start -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration (New-TimeSpan -Days 3650)
} catch {
    $t1 = New-ScheduledTaskTrigger -Once -At $start -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration (New-TimeSpan -Days 365)
}
$a = New-ScheduledTaskAction -Execute $pyw -Argument "lab_daemon.py" -WorkingDirectory $dir
Register-ScheduledTask -TaskName "btcvol-lab" -Action $a -Trigger $t1 -Settings $settings -Force `
    -Description "BTC alpha lab: 24/7 paper trading of every hypothesis (lab_daemon.py)" | Out-Null
Start-ScheduledTask -TaskName "btcvol-lab"
Write-Host ""
Write-Host "btcvol-lab registered and started. Report: reports\lab.html (refreshes every 15 min). Log: logs\lab_*.log"
