# The morning check of the laptop's night (scripts/night_check.py): every day at 09:30 local time, later if
# the laptop slept, without waking it and without a console window. Run once in PowerShell as the user:
#   powershell -NoProfile -ExecutionPolicy Bypass -File deploy\night-check-task.ps1
# Remove: Unregister-ScheduledTask -TaskName RBRY-NightCheck -Confirm:$false
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\pythonw.exe"
$action = New-ScheduledTaskAction -Execute $python -Argument "scripts\night_check.py" -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Daily -At "09:30"
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 5)
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName "RBRY-NightCheck" -Action $action -Trigger $trigger -Settings $settings `
    -Principal $principal -Description "RBRY: did the night's work run (scripts/night_check.py)" -Force | Out-Null
Get-ScheduledTask -TaskName "RBRY-NightCheck" | Get-ScheduledTaskInfo | Select-Object TaskName, NextRunTime
