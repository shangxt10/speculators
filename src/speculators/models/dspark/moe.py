"""Portable routed/shared SwiGLU experts for the existing DSpark attention stack."""

from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import nn
from torch.nn import functional


class DraftMoEConfig(BaseModel):
    """FFN shape and routing are independent of the verifier's attention type."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    num_experts: int = Field(default=64, gt=0)
    experts_per_token: int = Field(default=4, gt=0)
    expert_intermediate_size: int = Field(default=1024, gt=0)
    num_shared_experts: int = Field(default=1, ge=0)
    score_function: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "softmax"
    normalize_topk: bool = True
    routed_scaling_factor: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def validate_topk(self):
        if self.experts_per_token > self.num_experts:
            raise ValueError("experts_per_token cannot exceed num_experts")
        return self


class PackedExperts(nn.Module):
    """Stacked parameters avoid hundreds of FSDP parameter objects per layer."""

    def __init__(self, hidden_size, intermediate_size, num_experts):
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size, hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size)
        )
        nn.init.normal_(self.gate_up_proj, std=0.02)
        nn.init.normal_(self.down_proj, std=0.02)

    def forward(self, x, indices, weights):
        # Reference implementation: dispatch only selected token/expert pairs.
        # No dense [tokens, experts, hidden] tensor or expert-capacity dropping.
        output = torch.zeros_like(x)
        for expert in range(self.gate_up_proj.shape[0]):
            tokens, slots = torch.where(indices == expert)
            if tokens.numel() == 0:
                continue
            selected = x[tokens]
            gate, up = functional.linear(selected, self.gate_up_proj[expert]).chunk(
                2, dim=-1
            )
            value = functional.linear(
                functional.silu(gate) * up, self.down_proj[expert]
            )
            output.index_add_(
                0, tokens, value * weights[tokens, slots, None].to(value.dtype)
            )
        return output


class DSparkMoE(nn.Module):
    """Routed top-k experts plus optional always-active shared experts."""

    def __init__(self, hidden_size: int, config: DraftMoEConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(hidden_size, config.num_experts, bias=False)
        self.experts = PackedExperts(
            hidden_size, config.expert_intermediate_size, config.num_experts
        )
        self.shared_experts = (
            PackedExperts(
                hidden_size,
                config.expert_intermediate_size * config.num_shared_experts,
                1,
            )
            if config.num_shared_experts
            else None
        )
        nn.init.normal_(self.gate.weight, std=0.02)

    def forward(self, hidden_states):
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        if x.shape[0] == 0:
            return hidden_states
        scores = functional.linear(x.float(), self.gate.weight.float())
        if self.config.score_function == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.config.score_function == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = functional.softplus(scores).sqrt()
        weights, indices = scores.topk(self.config.experts_per_token, dim=-1)
        if self.config.normalize_topk:
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-20)
        weights = weights * self.config.routed_scaling_factor
        output = self.experts(x, indices, weights)
        if self.shared_experts is not None:
            gate, up = functional.linear(x, self.shared_experts.gate_up_proj[0]).chunk(
                2, dim=-1
            )
            output = output + functional.linear(
                functional.silu(gate) * up, self.shared_experts.down_proj[0]
            )
        return output.reshape(shape)

    def active_parameter_count(self):
        """Per-position FFN budget, including router and shared experts."""
        per_expert = (
            self.experts.gate_up_proj[0].numel() + self.experts.down_proj[0].numel()
        )
        shared = (
            sum(p.numel() for p in self.shared_experts.parameters())
            if self.shared_experts
            else 0
        )
        return (
            self.gate.weight.numel()
            + self.config.experts_per_token * per_expert
            + shared
        )
