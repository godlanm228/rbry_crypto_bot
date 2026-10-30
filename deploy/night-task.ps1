# The laptop's night for RBRY learning (scripts/night.py): every day at 00:00 local time, waking the laptop,
# at most 7 hours. Run once in PowerShell as the user (no admin needed):
#   powershell -NoProfile -ExecutionPolicy Bypass -File deploy\night-task.ps1
# Remove: Unregister-ScheduledTask -TaskName RBRY-Night -Confirm:$false
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$action = New-ScheduledTaskAction -Execute $python -Argument "scripts\night.py" -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Daily -At "00:00"
$settings = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 7)
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName "RBRY-Night" -Action $action -Trigger $trigger -Settings $settings `
    -Principal $principal -Description "RBRY: fresh data and the night's learning (scripts/night.py)" -Force | Out-Null
Get-ScheduledTask -TaskName "RBRY-Night" | Get-ScheduledTaskInfo | Select-Object TaskName, NextRunTime
