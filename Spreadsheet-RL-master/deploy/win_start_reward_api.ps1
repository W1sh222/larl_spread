# 启动 Spreadsheet-RL 奖励服务（依赖 Excel COM，必须跑在交互式会话里，不能在 SSH/Session 0 启动）
$ErrorActionPreference = 'Stop'
$root = 'E:\SpreadsheetRL'
New-Item -ItemType Directory -Force -Path "$root\db", "$root\recalc-jobs" | Out-Null

$env:REWARD_API_OUTPUT_ROOT = "$root\data"
$env:REWARD_API_RECALC_JOB_ROOT = "$root\recalc-jobs"
$env:REWARD_API_DB_PATH = "$root\db\jobs.sqlite3"
$env:REWARD_API_MAX_UPLOAD_BYTES = '104857600'
$env:REWARD_API_WINDOWS_EXCEL_RECYCLE_PRIVATE_MB = '4096'

Set-Location "$root\reward"
& "$root\reward\.venv\Scripts\python.exe" -m async_reward_api --platform windows --host 0.0.0.0 --port 5000 --workers 2 --instance-per-worker 2 *>> "$root\reward_api.log"
