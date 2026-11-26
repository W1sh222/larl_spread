# 环境兼容矩阵

本文件给出一组面向 **driver 570 / SM86** 目标机的推荐依赖组合，以及每个版本的选取依据。
本仓库 A100（SM80）集群上的实测版本见 [environment.md](environment.md)；两者只在
CUDA / vLLM / torch 轮子上不同，其余约束（sdpa、numpy 1.26、无 torchcodec、FSDP2 DTensor）一致。

## 1. 推荐组合

| 组件 | 版本 | 说明 |
| --- | --- | --- |
| Python | 3.11 | 仓库代码使用 `list[...]` / `X \| Y` 注解（需 ≥ 3.10）；3.11 的预编译轮子覆盖最全 |
| torch | 2.10.0+cu126 | CUDA 12.6 轮子，兼容 driver ≥ 525 的机器 |
| torchvision | 0.25.0 | 与 torch 2.10 官方配对 |
| vLLM | 0.17.0 | rollout 与评测共用的推理后端 |
| transformers | 5.20.2 | tokenizer / processor 接口与 vLLM 0.17 对齐 |
| numpy | 1.26.x | 固定 `< 2` |

## 2. 选取依据

### 2.1 CUDA / 驱动

driver 570 对应的 CUDA ceiling 是 12.8。装 `cu126` 轮子不需要升级驱动，也不占用
驱动升级窗口；`cu128` 轮子会把驱动要求顶到 570 以上，驱动一旦落后就起不来。
因此 torch 选 `2.10.0+cu126`，而不是更高 CUDA 版本的轮子。

### 2.2 GPU 架构（SM86）

SM86（A10 / A4000 级）没有 FlashAttention-3（仅 Hopper SM90 提供），推理与训练都必须
走 `sdpa` / FA2 路径：

- 训练侧 `deploy/run_grpo.sh` 默认 `attn_implementation=sdpa`；
- vLLM 侧不要开 CUDA graph 与 flashinfer 采样（`enforce_eager`、`VLLM_USE_FLASHINFER_SAMPLER=0`）。

### 2.3 FSDP2 / DTensor

verl 的 FSDP2 checkpoint 保存的是 DTensor 分片，merger 通过 DTensor 的 `placements`
合并 shard：

- torch 2.5 起 DTensor 位于 `torch.distributed.tensor`，旧路径 `torch.distributed._tensor`
  仍然可用；`verl/model_merger/fsdp_model_merger.py` 对两者都做了兼容 import。
- 不要降到 torch 2.4 之前：DTensor 序列化格式不一致，`python -m verl.model_merger merge`
  读不了 ckpt。torch 2.10 的 shard 格式与 `deploy/export_final_hf.sh` 使用的 merger 兼容。

### 2.4 vLLM ↔ torch ↔ transformers

- vLLM 的预编译轮子按特定 torch ABI 编译，torch 小版本错配会在 import 期报
  `undefined symbol`；vLLM 0.17.0 对应 torch 2.10.x。
- transformers 5.x 移除了部分 tokenizer / processor 旧接口，vLLM 0.17 已在 5.20.2 上对齐；
  退回 4.x 会走兼容分支，tokenize 结果可能与训练侧不一致。

### 2.5 torchvision / numpy

- torchvision 的 0.2x 版本与 torch 2.x 一一配对（0.25 ↔ 2.10）。错配会在
  `import torchvision` 时报 `operator torchvision::nms does not exist`。
- numpy 锁 1.26.x：2.x 改了 C API，未重编译的扩展会报 `_ARRAY_API not found`。

## 3. 不兼容组合

| 组合 | 症状 | 处理 |
| --- | --- | --- |
| `torchcodec==0.17` | vLLM 启动 `ImportError` | `pip uninstall -y torchcodec` |
| numpy ≥ 2 | `Failed to initialize NumPy: _ARRAY_API not found` | 降到 1.26.x |
| torch 2.10 + vLLM 0.16 及更早 | C++ 扩展 ABI 不匹配 | 按表 1 成对升级 |
| torch < 2.5 | FSDP2 DTensor ckpt 无法 merge | 升级 torch |
| flashinfer 采样 kernel | `--compress-mode=size` 不支持 | `VLLM_USE_FLASHINFER_SAMPLER=0` |

## 4. 与本仓库实测部署的差异

| 项 | 本表（driver 570 / SM86） | 实测部署（见 environment.md） |
| --- | --- | --- |
| GPU | SM86 | A100 80GB（SM80） |
| torch | 2.10.0+cu126 | 2.13.0+cu129 |
| vLLM | 0.17.0 | 0.32.0 |

迁移到别的硬件时，只改这三个包（torch / torchvision / vLLM）并保持 transformers、numpy
的约束不变；不要逐包单独升级，否则最容易踩 2.4 与 2.5 的 ABI 问题。