# 注册并启动 Spreadsheet-RL 奖励服务的交互式计划任务（Excel COM 需要在交互式会话）
$ErrorActionPreference = 'Stop'
$userId = "$env:COMPUTERNAME\$env:USERNAME"

$action = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument '-NoProfile -ExecutionPolicy Bypass -File E:\SpreadsheetRL\win_start_reward_api.ps1'

$principal = New-ScheduledTaskPrincipal -UserId $userId -LogonType Interactive -RunLevel Highest

$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName 'SRL-RewardAPI' -Action $action -Principal $principal `
    -Settings $settings -Description 'Spreadsheet-RL async reward API (Excel COM)' -Force | Out-Null

Start-ScheduledTask -TaskName 'SRL-RewardAPI'
Start-Sleep -Seconds 5
Get-ScheduledTask -TaskName 'SRL-RewardAPI' | Select-Object TaskName, State
