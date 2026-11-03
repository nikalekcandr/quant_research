# Removes the scheduled tasks created by install_tasks.ps1 and install_lab.ps1 (e.g. after moving to a server).
#   powershell -ExecutionPolicy Bypass -File uninstall_tasks.ps1
foreach ($name in @("btcvol-live", "btcvol-daily", "btcvol-fast", "btcvol-lab")) {
    if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $name -Confirm:$false
        Write-Host "removed $name"
    }
}
