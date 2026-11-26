# 启动 Spreadsheet-RL 奖励服务（依赖 Excel COM，必须跑在交互式会话里，不能在 SSH/Session 0 启动）
$ErrorActionPreference = 'Stop'
$root = 'E:\SpreadsheetRL'
New-Item -ItemType Directory -Force -Path "$root\db", "$root\recalc-jobs", "$root\tmp" | Out-Null

# Starlette 会把每个 >1MB 的上传体 spool 到 %TEMP%。默认在 C:，高并发时会写满
# 系统盘并触发 Defender 全盘扫描，整机卡死。这里统一切到数据盘。
$env:TEMP = "$root\tmp"
$env:TMP = "$root\tmp"

$env:REWARD_API_OUTPUT_ROOT = "$root\data"
$env:REWARD_API_RECALC_JOB_ROOT = "$root\recalc-jobs"
$env:REWARD_API_DB_PATH = "$root\db\jobs.sqlite3"
$env:REWARD_API_MAX_UPLOAD_BYTES = '104857600'
$env:REWARD_API_WINDOWS_EXCEL_RECYCLE_PRIVATE_MB = '3072'
$env:REWARD_API_WINDOWS_EXCEL_RECYCLE_JOBS = '50'

# 准入控制：单进程 4 个 Excel，队列只保留 256 条待办，同时最多处理 8 个上传。
# 超出直接拒绝，避免把上千个工作簿落盘后再排队。
$env:REWARD_API_MAX_QUEUE_SIZE = '256'
$env:REWARD_API_MAX_INFLIGHT_UPLOADS = '8'
$env:REWARD_API_STALE_QUEUED_S = '600'
$env:REWARD_API_WORKER_TIMEOUT_S = '240'

Set-Location "$root\reward"
& "$root\reward\.venv\Scripts\python.exe" -m async_reward_api --platform windows --host 0.0.0.0 --port 5000 --workers 1 --instance-per-worker 4 *>> "$root\reward_api.log"
