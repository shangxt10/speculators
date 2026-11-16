"""Optional training-only supervision of intermediate DSpark draft layers."""

from functools import partial

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn
from torch.nn import functional
from torch.utils.checkpoint import checkpoint

from speculators.models.dspark.metrics import _masked_decayed_mean
from speculators.models.metrics import dflash_loss_decay


class IntermediateSupervisionConfig(BaseModel):
    """Persisted recipe; decay uses the trainer's resumed optimizer-step count."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    rank: int = Field(default=256, gt=0)
    weight: float = Field(default=0.2, ge=0)
    decay_steps: int | None = Field(default=None, gt=0)
    weight_end: float = Field(default=0.08, ge=0)
    logit_chunk_size: int = Field(default=512, gt=0)

    def weight_at(self, step: int) -> float:
        if self.decay_steps is not None and step >= self.decay_steps:
            return self.weight_end
        return self.weight


class IntermediateHead(nn.Module):
    """RMSNorm -> low-rank projection -> independent vocabulary projection."""

    def __init__(self, hidden_size: int, rank: int, vocab_size: int, eps: float):
        super().__init__()
        self.eps = eps
        self.norm_weight = nn.Parameter(torch.ones(hidden_size))
        self.down = nn.Parameter(torch.empty(rank, hidden_size))
        self.codebook = nn.Parameter(torch.empty(vocab_size, rank))
        nn.init.normal_(self.down, std=hidden_size**-0.5)
        nn.init.normal_(self.codebook, std=rank**-0.5)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        x = hidden.flatten(0, 1)
        normalized = x.float() * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + self.eps
        )
        return functional.linear(normalized.to(x.dtype) * self.norm_weight, self.down)


def _chunk_ce(code, codebook, labels):
    return functional.cross_entropy(
        functional.linear(code, codebook).float(), labels, reduction="none"
    )


def auxiliary_loss(
    heads, states, targets, mask, block_size, gamma, config, sample_from_anchor
):
    """Use the final head's aligned teacher labels and exact padding/anchor mask."""
    labels = targets.detach().argmax(-1).flatten()
    positions = (torch.arange(labels.numel(), device=labels.device) % block_size)[None]
    decay = partial(
        dflash_loss_decay, gamma=gamma, sample_from_anchor=sample_from_anchor
    )
    total = targets.new_zeros((), dtype=torch.float32)
    metrics = {}
    for index, (head, hidden) in enumerate(zip(heads, states, strict=True), 1):
        code = head(hidden)
        chunk = config.logit_chunk_size
        ce = (
            torch.cat(
                [
                    checkpoint(
                        _chunk_ce,
                        code[i : i + chunk],
                        head.codebook,
                        labels[i : i + chunk],
                        use_reentrant=False,
                    )
                    for i in range(0, labels.numel(), chunk)
                ]
            )
            if labels.numel()
            else code.sum(-1).float()
        )
        term = _masked_decayed_mean(ce[None], mask, positions, decay)
        total = total + term
        metrics[f"inter{index}_ce_sum"] = term.detach()
        metrics[f"inter{index}_ce_total"] = term.new_ones(())
    return total, metrics
