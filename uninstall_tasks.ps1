# Removes the scheduled tasks created by install_tasks.ps1.
#   powershell -ExecutionPolicy Bypass -File uninstall_tasks.ps1
foreach ($name in @("btcvol-live", "btcvol-daily", "btcvol-fast")) {
    if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $name -Confirm:$false
        Write-Host "removed $name"
    }
}
