# StockIntel Daily Schedule Script
# Run this once to set up daily automation:
#   powershell -ExecutionPolicy Bypass -File stock_intel\data\schedule_daily.ps1
#
# This creates a Windows Task Scheduler task that runs the pipeline
# every trading day at 4:30 PM (after market close).

$TaskName = "StockIntelDaily"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$PythonPath = (Get-Command python).Source
$RunScript = Join-Path -Path $ProjectRoot -ChildPath "stock_intel\run.py"
$LogDir = Join-Path -Path $ProjectRoot -ChildPath "stock_intel\data\logs"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path -Path $LogDir -ChildPath "daily_$Timestamp.log"

# Create log directory
if (-not (Test-Path -LiteralPath $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

# Remove existing task if it exists
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed existing task: $TaskName"
}

# Create the scheduled task action
$Action = New-ScheduledTaskAction -Execute $PythonPath -Argument "-m stock_intel.run --full --verbose" -WorkingDirectory $ProjectRoot

# Run at 4:30 PM weekdays (Mon-Fri) and 10:00 AM weekends (for catch-up)
$TriggerWeekday = New-ScheduledTaskTrigger -Daily -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At 03:00PM
$TriggerWeekend = New-ScheduledTaskTrigger -Daily -DaysOfWeek Saturday,Sunday -At 10:00AM

# Run with highest privileges
$Principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest

# Settings
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 2)

# Register
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $TriggerWeekday -Principal $Principal -Settings $Settings -Description "StockIntel daily pipeline: scan -> score -> evaluate -> research -> dashboard"

Write-Host "`nTask '$TaskName' created successfully!"
Write-Host "Schedule: Weekdays at 4:30 PM, Weekends at 10:00 AM"
Write-Host "Working Directory: $ProjectRoot"
Write-Host "Logs: $LogDir"
Write-Host "`nTo run immediately: Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "To view: Get-ScheduledTask -TaskName '$TaskName' | fl"
