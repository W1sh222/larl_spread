# Windows 侧部署

Windows 主机承担两件事，**都依赖 Microsoft Excel COM，都必须在已登录的交互式桌面上运行**：

1. `SRL-RewardAPI`：GRPO 训练用的奖励 / 重算 HTTP 服务（端口 5000）
2. `SBT-Recalc`：SpreadsheetBench 的 GT 批量重算队列（按需触发）

本机实测环境：Windows 10 教育版，用户 `fyclab`（管理员），Office 16 (`C:\Program Files\Microsoft Office\Root\Office16\EXCEL.EXE`)，内部地址 `172.16.6.155`。

## 1. 部署位置

```
E:\SpreadsheetRL\
├── reward\                    # 奖励服务源码 + .venv
├── data\                      # 与 Linux 侧同构的数据副本（domain / excelforum / spreadsheetbench*）
├── db\jobs.sqlite3            # 作业状态
├── recalc-jobs\               # 重算临时工作区
├── reward_api.log
├── win_install_reward_task.ps1
└── win_start_reward_api.ps1

E:\SpreadsheetBench\
├── Python311\                 # 独立 Python 3.11
├── open_spreadsheet.py        # 单目录重算脚本（win32com 后端）
├── run_queue.ps1              # 队列驱动脚本
├── queue.txt / done.txt / recalc.log
└── work\
```

## 2. 安装 Python 环境

```powershell
# 独立 venv，不污染系统 Python
cd E:\SpreadsheetRL\reward
uv sync --locked          # 或  python -m venv .venv; .venv\Scripts\pip install -e .
.venv\Scripts\python.exe -V   # 实测 Python 3.11.9
```

`pywin32` 由 `async_reward_api` 的依赖带入，无需单独安装。

## 3. 数据同步

奖励服务要比对 `target.xlsx`，因此 Windows 上需要与 Linux 同构的数据目录。只同步**必要文件**（`instruction.json` + `target.xlsx` 等），不要同步整个 `_workspaces`：

```bash
# Linux → Windows（在 Linux 上执行）
tar -czf srl_meta.tgz -C Spreadsheet-RL-master/data \
    domain excelforum spreadsheetbench spreadsheetbench_2 spreadsheetbench_verified
scp srl_meta.tgz fyclab@172.16.6.155:E:/SpreadsheetRL/
# Windows 上解压到 E:\SpreadsheetRL\data
```

## 4. 启动奖励服务

**必须先注册交互式计划任务，不能通过 SSH 直接启动。**

```powershell
# 在 Windows 上（管理员 PowerShell）
powershell -ExecutionPolicy Bypass -File E:\SpreadsheetRL\win_install_reward_task.ps1
```

脚本内容等价于：

```powershell
$userId = "$env:COMPUTERNAME\$env:USERNAME"
$action = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument '-NoProfile -ExecutionPolicy Bypass -File E:\SpreadsheetRL\win_start_reward_api.ps1'
$principal = New-ScheduledTaskPrincipal -UserId $userId -LogonType Interactive -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName 'SRL-RewardAPI' -Action $action -Principal $principal `
    -Settings $settings -Description 'Spreadsheet-RL async reward API (Excel COM)' -Force
Start-ScheduledTask -TaskName 'SRL-RewardAPI'
```

服务健康检查（Linux 侧）：

```bash
curl -s http://172.16.6.155:5000/health
```

主要接口：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查 |
| POST | `/reward/submit` | 上传工作簿，返回 `job_id`（异步） |
| GET | `/reward/result/{job_id}` | 轮询评分结果 |
| GET | `/reward/status/{job_id}` | 只查状态 |
| POST | `/recalculate` | 同步重算（工具链内部使用） |
| POST | `/recalculate/submit` + `/recalculate/status/{job_id}` | 异步重算 |

并发：`--workers 2 --instance-per-worker 2` = 4 个常驻 Excel 实例。OOM 时降 `--workers` 或设 `REWARD_API_MAX_RUNNING_JOBS`。

## 5. SBT-Recalc（评测 GT 批量重算）

用于把一批目录里的 xlsx 全部重算一遍（例如 GT 只存公式、需补齐缓存值）。

```powershell
# 把待重算目录逐行写入队列
'D:\data\spreadsheet\task-001' | Set-Content E:\SpreadsheetBench\queue.txt -Encoding UTF8
'D:\data\spreadsheet\task-002' | Add-Content E:\SpreadsheetBench\queue.txt -Encoding UTF8

# 触发（脚本会先删 done.txt，跑完再写回，可用它判断结束）
Start-ScheduledTask -TaskName 'SBT-Recalc'
# 等待 E:\SpreadsheetBench\done.txt 出现
```

失败不中断：单个目录报错只写日志，继续下一个。

## 6. SSH 访问（给别人用的密钥）

### 6.1 密钥位置

```
/remote-home/share/yichengfu/larl_spread/ssh_key/
├── id_ed25519        # 私钥（600，仅 root 可读）
└── id_ed25519.pub    # 公钥
```

公钥内容：

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHgPrILRZzRfbZ15LEuCy8cUKyfg54JzkQBmP1PUfq89 larl_spread-windows-access
```

该公钥**已导入** Windows 的 `C:\ProgramData\ssh\administrators_authorized_keys`，持有对应私钥即可免密登录。

### 6.2 使用

```bash
chmod 600 id_ed25519
ssh -i ./id_ed25519 fyclab@172.16.6.155
# 验证
ssh -i ./id_ed25519 fyclab@172.16.6.155 'echo OK; hostname'
```

默认 shell 是 PowerShell，可直接执行 PowerShell 命令；文件传输用 `scp -i ./id_ed25519`。

### 6.3 给别人用之前必须知道

- 私钥当前权限 600 / 属主 root。要交付给他人，请**复制一份**并 `chown` 给对方，或让对方自己生成密钥、把公钥发给你追加到 `administrators_authorized_keys`（更推荐，便于单独吊销）。
- 该密钥对应的是 **Windows 管理员账户 `fyclab`**，即拥有目标机器完整管理员权限。不要把私钥放到公开仓库、聊天群或共享目录的 world-readable 位置。
- 吊销：编辑 `C:\ProgramData\ssh\administrators_authorized_keys` 删除对应行即可，无需重启服务。
- Linux 侧 `~/.ssh/id_ed25519`（`larl-agent`）是另一把**旧的**运维密钥，与这把无关。

### 6.4 Windows SSH 服务（重装时参考）

```powershell
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Set-Service sshd -StartupType Automatic
Start-Service sshd
# 若 PowerShell 作为默认 shell
New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell `
    -Value "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" -PropertyType String -Force
```

管理员账号的公钥文件是 `C:\ProgramData\ssh\administrators_authorized_keys`（**不是** `~/.ssh/authorized_keys`），且必须修正 ACL 否则不生效：

```powershell
icacls C:\ProgramData\ssh\administrators_authorized_keys /inheritance:r `
    /grant "Administrators:F" /grant "SYSTEM:F"
```

## 7. 硬约束

- **Windows 必须保持登录状态**：锁屏可以，注销 / 关机 / 重启后计划任务没有交互式会话，Excel COM 全部失败 → 训练 reward 全 0。
- 重启后手工确认：`Get-ScheduledTask -TaskName SRL-RewardAPI | Select State` 应为 `Running`，不正确时 `Start-ScheduledTask`。
- 防火墙放行 5000（内网）：

```powershell
New-NetFirewallRule -DisplayName "SRL-Reward-5000" -Direction Inbound -Protocol TCP -LocalPort 5000 -Action Allow
```

- `OpenSSH-Server-In-TCP` 规则由系统自动创建，重复创建会报"已存在"，属正常。
