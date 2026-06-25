# StockIntel Schedule Setup
#   powershell -ExecutionPolicy Bypass -File stock_intel\data\schedule_daily.ps1
#
# Runs the full pipeline Tue-Sat at 2:00 AM IST (= Mon-Fri 4:30 PM ET market close)

$ErrorActionPreference = "Stop"
$TaskName = "StockIntelDaily"
$Python = "C:\Users\Abeer sharma\AppData\Local\Programs\Python\Python314\python.exe"
$WorkDir = "E:\Work\stockscanner"

# Remove existing task if present
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed existing task: $TaskName"
}

# Create action
$Action = New-ScheduledTaskAction -Execute $Python `
    -Argument "-m stock_intel.run --schedule" `
    -WorkingDirectory $WorkDir

# Trigger: Tue-Sat at 2:00 AM IST (covers Mon-Fri US market close)
$Trigger = New-ScheduledTaskTrigger -Weekly `
    -DaysOfWeek Tuesday, Wednesday, Thursday, Friday, Saturday `
    -At 02:00AM

# Run as SYSTEM with highest privileges
$Principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest

# Settings: 2-hour timeout, start on batteries, dont stop on batteries
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -MultipleInstances IgnoreNew

# Register
Register-ScheduledTask -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Principal $Principal `
    -Settings $Settings `
    -Description "StockIntel daily pipeline: scan, score, evaluate, research, email" `
    -Force

Write-Host "Task '$TaskName' created successfully!"
Write-Host "Schedule: Tue-Sat at 2:00 AM IST (Mon-Fri 4:30 PM ET market close)"
Write-Host "To run now: Start-ScheduledTask -TaskName '$TaskName'"
