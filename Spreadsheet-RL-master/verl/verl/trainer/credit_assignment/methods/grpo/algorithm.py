# GRPO 作为 Credit Assignment Library 的一个 method: 复用 core_algos 的既有实现,
# 不重写公式. 唯一职责是把 CreditAssignmentInput 还原成 vanilla 函数需要的 token-level 输入.

from typing import Any

import numpy as np
import torch

from verl.trainer.credit_assignment.interface import CreditAssignmentInput, CreditAssignmentMethod, CreditAssignmentOutput
from verl.trainer.credit_assignment.methods.grpo.config import GRPOConfig
from verl.trainer.credit_assignment.registry import register_credit_assignment


def _reward_dtype(trajectory_rewards: np.ndarray, assistant_mask: torch.Tensor) -> torch.dtype:
    """避免用 zeros_like(mask) 决定 dtype: bool/int mask 会把 reward 截断."""
    probe = torch.as_tensor(np.asarray(trajectory_rewards)).dtype
    if probe.is_floating_point:
        return probe
    if assistant_mask.dtype.is_floating_point:
        return assistant_mask.dtype
    return torch.float32


@register_credit_assignment("grpo")
class GRPO(CreditAssignmentMethod):
    name = "grpo"
    # 只用 trajectory reward / assistant mask / group id, 不依赖任何 step 级 metadata.
    requires = frozenset()

    def compute(self, inputs: CreditAssignmentInput, params: dict[str, Any]) -> CreditAssignmentOutput:
        cfg = GRPOConfig.from_params(params)
        # 延迟 import: core_algos 依赖 trainer 栈, 顶层 import 会与 registry/methods 形成环.
        from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

        mask = inputs.assistant_mask
        rewards = np.asarray(inputs.trajectory_rewards)
        if rewards.shape[0] != mask.shape[0]:
            raise ValueError(f"GRPO: trajectory_rewards rows {rewards.shape[0]} != assistant_mask rows {mask.shape[0]}")

        dtype = _reward_dtype(rewards, mask)
        token_level_rewards = _place_on_last_valid_token(
            rewards, mask, dtype=dtype, device=mask.device
        )

        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=mask,
            index=inputs.episode_group_ids,
            epsilon=cfg.epsilon,
            norm_adv_by_std_in_grpo=cfg.norm_adv_by_std_in_grpo,
        )

        return CreditAssignmentOutput(
            advantages=advantages, returns=returns, metrics=_diagnostics(advantages, mask, inputs.episode_group_ids)
        )


def _place_on_last_valid_token(
    trajectory_rewards: np.ndarray, mask: torch.Tensor, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """reward 落在行内最后一个有效 assistant token; 全零行退回位置 0, 保证 sum(-1) 守恒."""
    rewards = torch.as_tensor(np.asarray(trajectory_rewards), dtype=dtype, device=device).reshape(-1)
    out = torch.zeros(mask.shape, dtype=dtype, device=device)
    lengths = (mask != 0).sum(dim=-1)
    pos = torch.where(lengths > 0, lengths - 1, torch.zeros_like(lengths)).to(torch.long)
    out.scatter_(1, pos.unsqueeze(-1), rewards.unsqueeze(-1))
    return out


def _diagnostics(advantages: torch.Tensor, mask: torch.Tensor, group_ids: np.ndarray) -> dict[str, float]:
    adv = advantages.detach().float().cpu().numpy()
    valid = (mask != 0).detach().cpu().numpy()
    flat = adv[valid] if valid.any() else np.zeros(0, dtype=np.float32)
    groups: dict[Any, int] = {}
    for gid in np.asarray(group_ids, dtype=object):
        groups[gid] = groups.get(gid, 0) + 1
    sizes = np.asarray(list(groups.values()), dtype=np.float64) if groups else np.zeros(0)
    return {
        "adv_mean": float(flat.mean()) if flat.size else 0.0,
        "adv_std": float(flat.std()) if flat.size else 0.0,
        "zero_adv_ratio": float((np.abs(flat) < 1e-8).mean()) if flat.size else 0.0,
        "num_groups": float(len(groups)),
        "singleton_group_ratio": float((sizes == 1).mean()) if sizes.size else 0.0,
    }
