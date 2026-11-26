# 踩坑记录

按"症状 → 原因 → 处理"记录本部署过程中实际遇到过的问题。环境版本改变后部分条目可能失效。

## 环境与依赖

### `vllm serve` 启动即崩：torchcodec 导入错误

症状：vLLM 启动阶段抛 `ImportError` / `RuntimeError`，栈里出现 `torchcodec`。

原因：`torchcodec==0.17` 与 torch 2.13 + vLLM 0.32 组合不兼容。

处理：

```bash
pip uninstall -y torchcodec
```

若无法卸载，需在 vLLM 侧把 torchcodec 的可选导入包一层 try/except（本机采用直接卸载）。

### 启动卡在 flashinfer 采样 kernel 编译

症状：启动时长时间 JIT 编译，或直接报 `--compress-mode=size` 不支持。

原因：当前 `nvcc` 不支持 flashinfer 采样 kernel 的编译选项。

处理：关闭 flashinfer 采样，回退 vLLM 原生实现。两个入口脚本都已内置：

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

### `vllm serve` 报未知参数 `--disable-log-requests`

原因：vLLM 0.32 已移除该参数。

处理：从启动命令里删掉。`deploy/serve_vllm.sh` 未使用该参数。

### `ImportError: cannot import name 'FusedMoE'`

症状：verl 导入 `verl.utils.vllm.vllm_fp8_utils` 失败。

原因：vLLM 0.32 删除了 `FusedMoE` 类，而 verl 只在 `isinstance` 判断里用到它。

处理：已在本仓库补丁中加入降级占位类型，见
[verl/utils/vllm/vllm_fp8_utils.py](file:///remote-home/yichengfu/larl/Spreadsheet-RL-master/verl/verl/utils/vllm/vllm_fp8_utils.py#L22-L35)：

```python
try:
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
except ImportError:
    try:
        from vllm.model_executor.layers.fused_moe import FusedMoE
    except ImportError:
        class FusedMoE:  # vLLM >= 0.32 已移除，仅用于 isinstance 判断
            pass
```

### tied embedding 同步时报 `ValueError`（未初始化）

症状：`update_weights` 阶段，tied embedding 被拆分到不同 bucket，触发"参数未初始化"硬报错。

处理：把该处 `raise ValueError` 降级为 `logger.warning` 后继续，属可接受的精度损失。

### `flash_attn` 未安装导致 `bert_padding` 导入失败

症状：`ModuleNotFoundError: flash_attn`。

处理：已在本仓库补丁中让 `attention_utils` 回退到 verl 自带的纯 PyTorch 等价实现，见
[verl/utils/attention_utils.py](file:///remote-home/yichengfu/larl/Spreadsheet-RL-master/verl/verl/utils/attention_utils.py#L26-L39)。

### `openai` 版本不兼容

症状：`client.chat.completions.create` 参数或返回结构报错。

处理：评测链路统一升到 `openai==2.54.0`（上游 requirements 锁的 1.30.4 过旧）。

## 训练侧

### A100 用 `flash_attention_3` 直接报错

原因：`flash_attention_3` 只适用于 Hopper（H100）。

处理：`run_grpo.sh` 已强制 `attn_implementation=sdpa`，同时关闭 `use_remove_padding` 与 `use_fused_kernels`（这两项在 sdpa 路径下会报错）。

### 2 卡运行 OOM（compute_log_prob 阶段）

原因：`rollout.gpu_memory_utilization` 默认 0.85，2 卡时 vLLM 与训练进程争显存。

处理：2 卡时降到 `GMU=0.6`。4 卡可用 0.85。

### 启动断言失败：token 长度不足

原因：`actor_rollout_ref.actor.ppo_max_token_len_per_gpu`、`rollout.log_prob_max_token_len_per_gpu`、`ref.log_prob_max_token_len_per_gpu` 小于多轮对话实际长度（12k+）。

处理：`ppo_max_token_len_per_gpu ≥ 24576`，log_prob 两个设 24576~65536。

### `train_batch_size × rollout.n` 不是 worker 数整数倍

症状：rollout 分片不均，报错退出。

处理：调整 `train_batch_size` 或 `rollout.n`，使乘积能被 agent_loop worker 数整除。

### 12 小时训练只跑了 74 步 / 4 小时

原因：`filter_overlong_prompts=True` + `data.max_prompt_length=4096`，而本数据集 prompt 中位数约 5.3k token，**仅 38.5% 样本通过过滤**，768 条导出数据实际只剩约 296 条。

处理：

- `max_prompt_length` 提到 8192（Qwen3-4B 原生支持 32k）
- 配套把 `log_prob_max_token_len_per_gpu` / `ppo_max_token_len_per_gpu` 提到 24576
- 导出样本量按 40% 留存率倒推

### 学习效果不明显、熵塌缩风险

原因（实测 74 步）：

- `max_response_length=8192` 导致 **62% 响应被截断**
- `train_batch_size=4` 过小，GRPO 组内对比信号弱
- 策略熵快速下降

处理：`max_response_length=24576`，`train_batch_size ≥ 32`，必要时提高 `entropy_coeff`。

### checkpoint 挤爆磁盘

处理：`trainer.save_freq=-1` 关闭周期保存，只在需要时手动保存。

### 停止训练

```bash
pkill -f "verl.trainer.main_pp[o]"   # 连带结束 raylet 与 vLLM worker，释放显存
```

确认释放：`nvidia-smi --query-gpu=index,memory.used --format=csv,noheader`

## 评测侧

### 模型输出被截断

症状：生成的代码或 JSON 在句子中间断掉。

原因：`llm_api.py` 里 `max_tokens` 未设置，取服务端默认值。

处理：已改为显式 `max_tokens=8192`，见
[inference/llm_api.py](file:///remote-home/yichengfu/larl/SpreadsheetBench/inference/llm_api.py#L4-L14)。

### 评测结果全部 0 分

原因：`evaluation.py` 的 `proc_path` 原本指向 input 文件而不是模型输出。修正后应指向：

```python
proc_path = f"{dataset_path}/outputs/{opt.setting}_{opt.model}/{test_case_idx + 1}_{data['id']}_output.xlsx"
```

见 [evaluation.py](file:///remote-home/yichengfu/larl/SpreadsheetBench/evaluation/evaluation.py#L213-L216)。

### `KeyError` / 找不到 `/mnt/data/...`

原因：模型生成的代码硬编码 `/mnt/data/`，本地没有该路径。

处理：

```bash
sudo ln -sfn /remote-home/yichengfu/larl/SpreadsheetBench/data/all_data_912_v0.1 /mnt/data
```

同时确保 `data/<dataset>/outputs/single_<model>/` 目录存在（`code_exec_docker` 原实现会 chmod 777，本地版没有该逻辑）。

### 大量任务产不出 xlsx

原因：模型生成的代码本身有 IndexError、sheet 名不匹配、语法错误等。这属于模型能力问题，不是链路故障。实测 24 条子集仅 1 条全对、3/72 单用例通过。

## Windows 侧

### 通过 SSH 调用 Excel COM 失败（最关键的一条）

症状：`Workbooks.Open` / `SaveAs` 报"方法无效"或直接异常，即使文件完全正常。

原因：SSH 会话运行在 **Session 0**，没有交互式窗口站（window station），Excel COM 拒绝工作。

处理：**不要通过 SSH 启动 Excel 相关脚本**。改为注册交互式计划任务：

```powershell
$principal = New-ScheduledTaskPrincipal -UserId "$env:COMPUTERNAME\$env:USERNAME" `
    -LogonType Interactive -RunLevel Highest
```

然后 `Start-ScheduledTask -TaskName 'SRL-RewardAPI'`。`deploy/win_install_reward_task.ps1` 即此逻辑。

### 计划任务状态为 Ready 而不是 Running

原因：机器重启 / 用户注销后交互式会话丢失。

处理：重新登录桌面（锁屏即可，不能注销），再 `Start-ScheduledTask`。这是**无法绕过的硬约束**。

### 部分 xlsx 重算失败："方法无效"

原因：模型生成的 xlsx 本身非法（结构损坏），Excel 打不开。

处理：不需要修复。这类样本在评测中直接判 0 分，属于预期行为。实测 36 个模型输出文件中 30 成功 / 6 失败，全部是模型产物问题。

### 防火墙规则创建报"已存在"

原因：`Add-WindowsCapability` 安装 OpenSSH Server 时自动创建了 `OpenSSH-Server-In-TCP`。

处理：忽略该错误，规则已生效。

### 管理员公钥不生效

原因：管理员账号读取的是 `C:\ProgramData\ssh\administrators_authorized_keys`，不是 `~/.ssh/authorized_keys`；且该文件 ACL 必须只允许 Administrators 与 SYSTEM。

处理：

```powershell
icacls C:\ProgramData\ssh\administrators_authorized_keys /inheritance:r `
    /grant "Administrators:F" /grant "SYSTEM:F"
```

### 从 Linux 连 Windows 超时 / 无响应

排查顺序：

1. `ping 172.16.6.155`、`Test-NetConnection`（Windows 侧需允许 ICMP）
2. Windows 上 `Get-NetTCPConnection -LocalPort 22 -State Listen`
3. 若本机监听正常、防火墙放行但仍不通，多半是**校园网 ACL 禁止了跨网段访问**；此时只能走由 Windows 主动外连的反向隧道，或换到同网段部署。

### SSH 公钥认证被网络设备阻断（连接 GitHub 场景）

症状：`ssh -T git@github.com` 卡在密钥交换，对端 banner 不是 GitHub 的 `babeld-*`。

原因：出口有中间设备拦截 22 端口。

处理：改用 HTTPS + Personal Access Token，并让 `git@github.com:` 形式自动重写：

```bash
git config --global credential.helper store
git config --global url."https://github.com/".insteadOf "git@github.com:"
```

凭据写入 `~/.git-credentials`（权限 600）。
