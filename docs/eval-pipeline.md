# 评测链路：SpreadsheetBench

链路：vLLM 推理 → 生成本地代码 → 本地代码执行服务改 xlsx → OJ 式逐用例比对打分。

## 0. 前置

| 服务 | 端口 | 启动脚本 |
| --- | --- | --- |
| vLLM（Qwen3.5-9B） | 8000 | `SpreadsheetBench/deploy/serve_vllm.sh` |
| 本地代码执行 | 8081 | `SpreadsheetBench/deploy/code_exec_local.py` |

`/mnt/data` 必须指向数据集目录（推理脚本生成的代码里硬编码该路径）：

```bash
sudo ln -sfn /remote-home/yichengfu/larl/SpreadsheetBench/data/all_data_912_v0.1 /mnt/data
```

## 1. 起服务

```bash
# 终端 1：vLLM（GPU 4，TP=1）
PORT=8000 GPU=4 bash SpreadsheetBench/deploy/serve_vllm.sh
# 看到 Uvicorn running on http://0.0.0.0:8000 即可

# 终端 2：本地代码执行（替代需要 Docker 的 code_exec_docker）
/remote-home/yichengfu/anaconda3/envs/spread/bin/python \
    SpreadsheetBench/deploy/code_exec_local.py --port 8081
```

服务地址自检：

```bash
curl http://127.0.0.1:8000/v1/models
curl -X POST http://127.0.0.1:8081/execute -H 'Content-Type: application/json' \
     -d '{"convid":"t","code":"print(1+1)"}'
```

## 2. 推理

```bash
cd SpreadsheetBench/inference
python inference_single.py \
    --model Qwen3.5-9B \
    --base_url http://127.0.0.1:8000/v1 \
    --api_key EMPTY \
    --dataset all_data_912_v0.1 \
    --code_exec_url http://127.0.0.1:8081/execute \
    --row 5
```

说明：

- `--dataset` 是 `data/` 下的目录名；只想试跑时用自己构造的子集目录（本部署曾用 `subset2`、`subset_test`）。
- 输出写入 `data/<dataset>/outputs/conv_single_<model>.jsonl` 与 `outputs/single_<model>/*.xlsx`。
- `--row` 是喂给模型的前 N 行预览，默认 5。
- 模型生成代码后立即在本地执行，**产出 xlsx 才可能得分**；代码报错即该条为 0 分。

## 3. 评测

```bash
cd SpreadsheetBench/evaluation
python evaluation.py --model Qwen3.5-9B --setting single --dataset all_data_912_v0.1
```

输出 `SpreadsheetBench/outputs/eval_single_<model>.json`，每条含：

- `test_case_results`：3 个测试用例各自的 0/1
- `soft_restriction`：用例通过率
- `hard_restriction`：全通过才为 1（即 Pass@1）

## 4. 只看汇总

```bash
python - <<'PY'
import json
r = json.load(open('SpreadsheetBench/outputs/eval_single_Qwen3.5-9B.json'))
n = len(r)
print(f"tasks={n}")
print(f"Pass@1(hard)={sum(x['hard_restriction'] for x in r)/n:.2%}")
print(f"soft={sum(x['soft_restriction'] for x in r)/n:.2%}")
PY
```

## 5. 与 Windows 的关系

评测链路**不需要 Windows**：比对的是本地 xlsx 的缓存值，不做公式重算。

例外情况：如果 GT 答案里的单元格只有公式没有缓存值，逐格比对会失败。这类 GT 需要先在 Windows 上重算一遍（`SBT-Recalc` 计划任务），这是训练侧 `reward` 服务的职责范围，评测侧不做处理。

## 6. 常见失败模式

| 现象 | 原因 |
| --- | --- |
| 大量任务无 output.xlsx | 模型代码报错（IndexError、sheet 名不匹配、语法错误），本地服务返回 traceback 到对话里但没有产物 |
| `KeyError: '/mnt/data/...'` | `/mnt/data` 软链接没建或指向错 |
| 推理输出被截断 | `max_tokens` 未设置，见 [troubleshooting](troubleshooting.md#评测侧) |
| 评测全部 0 分 | `evaluation.py` 的 `proc_path` 指到了 input 而非 output，见 [troubleshooting](troubleshooting.md#评测侧) |
