$ErrorActionPreference = 'Stop'
$filter = @{LogName='System';ProviderName='Microsoft-Windows-Power-Troubleshooter';Id=1;StartTime=[datetime]'2026-11-08'}
$rows = foreach ($event in Get-WinEvent -FilterHashtable $filter) {
    $fields = ([xml]$event.ToXml()).Event.EventData.Data
    $sleep = ($fields | Where-Object Name -eq 'SleepTime').'#text'
    $wake = ($fields | Where-Object Name -eq 'WakeTime').'#text'
    if ($sleep -and $wake) { [pscustomobject]@{sleep=$sleep;wake=$wake} }
}
$rows | Export-Csv -NoTypeInformation -Encoding utf8 -Path (Join-Path $PSScriptRoot '../data/raw/wing_gap_candles/sleep.csv')
