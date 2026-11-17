"""Opt-in Qwen3 multi-head latent attention for draft models.

KV content shares a low-rank bottleneck; a separate, shared key carries RoPE.
The cache stores only that rotated key and the normalized KV latent. Attention
currently expands the latent for the existing eager/SDPA/flex kernels (no fused
MLA decode kernel). See https://arxiv.org/abs/2405.04434, section 2.1.
"""

import copy

import torch
from torch import nn
from torch.nn import functional
from transformers import PretrainedConfig
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    eager_attention_forward,
)

from speculators.models.attention import ALL_ATTENTION_FUNCTIONS

MLA_DIMENSIONS = (
    "mla_kv_lora_rank",
    "mla_q_lora_rank",
    "mla_qk_nope_head_dim",
    "mla_qk_rope_head_dim",
    "mla_v_head_dim",
)


def layer_uses_mla(config: PretrainedConfig, layer_idx: int) -> bool:
    """Explicit MLA layer labels take precedence over the legacy global flag."""
    layer_types = getattr(config, "layer_types", None) or []
    if "mla" in layer_types:
        return layer_types[layer_idx] == "mla"
    return getattr(config, "use_mla", False)


def all_layers_use_mla(config: PretrainedConfig) -> bool:
    return all(layer_uses_mla(config, i) for i in range(config.num_hidden_layers))


def configure_mla(config: PretrainedConfig, **kwargs) -> None:
    """Resolve and persist MLA settings without touching legacy configurations."""
    if kwargs.get("draft_use_mla", False):
        config.use_mla = True
    overrides = {
        name: kwargs[name] for name in MLA_DIMENSIONS if kwargs.get(name) is not None
    }
    layer_types = getattr(config, "layer_types", None) or []
    if "mla" in layer_types and len(layer_types) != config.num_hidden_layers:
        raise ValueError("layer_types length must equal num_hidden_layers")
    if not getattr(config, "use_mla", False) and "mla" not in layer_types:
        if overrides:
            raise ValueError(
                "MLA dimensions require layer_types containing 'mla', "
                "--draft-use-mla or use_mla=true"
            )
        return
    if config.model_type != "qwen3":
        raise ValueError("MLA draft attention requires draft model_type='qwen3'")
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    rope_dim = max(2, (head_dim // 2) // 2 * 2)
    defaults = {
        "mla_kv_lora_rank": min(512, max(1, config.hidden_size // 4)),
        "mla_q_lora_rank": None,
        "mla_qk_nope_head_dim": max(1, head_dim - rope_dim),
        "mla_qk_rope_head_dim": rope_dim,
        "mla_v_head_dim": head_dim,
    }
    for name, default in defaults.items():
        value = overrides.get(name, getattr(config, name, default))
        if value is None and name == "mla_q_lora_rank":
            setattr(config, name, None)
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
        setattr(config, name, value)
    if config.mla_qk_rope_head_dim % 2:
        raise ValueError("mla_qk_rope_head_dim must be even")


def mla_rotary_config(config: PretrainedConfig) -> PretrainedConfig:
    """Generate frequencies for the decoupled RoPE dimension, not the full head."""
    rotary_config = copy.deepcopy(config)
    rotary_config.head_dim = config.mla_qk_rope_head_dim
    rotary_config.partial_rotary_factor = 1.0
    for name in ("rope_parameters", "rope_scaling"):
        params = getattr(rotary_config, name, None)
        if isinstance(params, dict) and "partial_rotary_factor" in params:
            params["partial_rotary_factor"] = 1.0
    return rotary_config


def _rotate(x, cos, sin, concat) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return x * cos + concat((-second, first), dim=-1) * sin


class Qwen3MLAAttention(nn.Module):
    """Qwen3 draft attention with optional DFlash target-context injection.

    ``input_size`` supports EAGLE's concatenated first-layer input. DFlash passes
    ``target_hidden`` to prepend target features to the draft's KV sequence.
    """

    def __init__(
        self, config: PretrainedConfig, layer_idx: int, input_size: int | None = None
    ):
        super().__init__()
        configure_mla(config)
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_groups = 1
        self.nope_dim = config.mla_qk_nope_head_dim
        self.rope_dim = config.mla_qk_rope_head_dim
        self.v_head_dim = config.mla_v_head_dim
        self.head_dim = self.nope_dim + self.rope_dim
        self.kv_lora_rank = config.mla_kv_lora_rank
        # Mixed models share ordinary Qwen3 rotary tables outside the layers.
        # Their MLA layers need separate frequencies for the decoupled branch.
        self.rotary_emb = (
            None
            if all_layers_use_mla(config)
            else Qwen3RotaryEmbedding(mla_rotary_config(config))
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        # All draft families construct their own causal/block/document masks.
        self.is_causal = False
        layer_types = getattr(config, "layer_types", None)
        self.sliding_window = (
            config.sliding_window
            if layer_types and layer_types[layer_idx] == "sliding_attention"
            else None
        )
        input_size = input_size or config.hidden_size
        bias = config.attention_bias
        if config.mla_q_lora_rank is None:
            self.q_proj = nn.Linear(
                input_size, self.num_heads * self.head_dim, bias=bias
            )
        else:
            self.q_a_proj = nn.Linear(input_size, config.mla_q_lora_rank, bias=bias)
            self.q_a_layernorm = Qwen3RMSNorm(
                config.mla_q_lora_rank, eps=config.rms_norm_eps
            )
            self.q_b_proj = nn.Linear(
                config.mla_q_lora_rank, self.num_heads * self.head_dim, bias=False
            )
        self.kv_a_proj_with_mqa = nn.Linear(
            input_size, self.kv_lora_rank + self.rope_dim, bias=bias
        )
        self.kv_a_layernorm = Qwen3RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.nope_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim, config.hidden_size, bias=bias
        )

    # Inference backends can replace concatenation without duplicating MLA math.
    concat_tensors = staticmethod(torch.cat)

    def forward(  # noqa: PLR0917
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask=None,
        past_key_values: Cache | None = None,
        cache_position: torch.Tensor | None = None,
        target_hidden: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, q_len, _ = hidden_states.shape
        if self.config.mla_q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        q_nope, q_rope = q.split([self.nope_dim, self.rope_dim], dim=-1)
        kv_input = (
            hidden_states
            if target_hidden is None
            else self.concat_tensors([target_hidden, hidden_states], dim=1)
        )
        latent, k_rope = self.kv_a_proj_with_mqa(kv_input).split(
            [self.kv_lora_rank, self.rope_dim], dim=-1
        )
        latent = self.kv_a_layernorm(latent).unsqueeze(1)
        k_rope = k_rope.unsqueeze(1)
        if self.rotary_emb is not None:
            if position_ids is None:
                raise ValueError("Mixed MLA layers require explicit position_ids")
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
        cos, sin = (part.unsqueeze(1) for part in position_embeddings)
        q_rope = _rotate(
            q_rope, cos[..., -q_len:, :], sin[..., -q_len:, :], self.concat_tensors
        )
        k_rope = _rotate(k_rope, cos, sin, self.concat_tensors)
        if past_key_values is not None:
            # Both tensors have one shared head. Never put expanded K/V in cache.
            k_rope, latent = past_key_values.update(
                k_rope,
                latent,
                self.layer_idx,
                {"cos": cos, "sin": sin, "cache_position": cache_position},
            )
        kv_len = latent.shape[-2]
        kv = (
            self.kv_b_proj(latent.squeeze(1))
            .view(bsz, kv_len, self.num_heads, self.nope_dim + self.v_head_dim)
            .transpose(1, 2)
        )
        k_nope, v = kv.split([self.nope_dim, self.v_head_dim], dim=-1)
        q = self.concat_tensors([q_nope, q_rope], dim=-1)
        k = self.concat_tensors(
            [k_nope, k_rope.expand(-1, self.num_heads, -1, -1)], dim=-1
        )
        implementation = self.config._attn_implementation or "eager"  # noqa: SLF001
        if implementation == "simple_flex_attention":
            # Flex requires supported power-of-two head widths. Zero-padding
            # preserves dot products; retain the original attention scale.
            width = 1 << (max(self.head_dim, self.v_head_dim) - 1).bit_length()
            q = functional.pad(q, (0, width - self.head_dim))
            k = functional.pad(k, (0, width - self.head_dim))
            v = functional.pad(v, (0, width - self.v_head_dim))
        elif (
            implementation == "eager"
            and attention_mask is not None
            and attention_mask.dtype == torch.bool
        ):
            attention_mask = torch.zeros_like(
                attention_mask, dtype=q.dtype
            ).masked_fill(~attention_mask, float("-inf"))
        attn_fn = (
            eager_attention_forward
            if implementation == "eager"
            else ALL_ATTENTION_FUNCTIONS[implementation]
        )
        output, weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        output = output[..., : self.v_head_dim].reshape(bsz, q_len, -1)
        return self.o_proj(output), weights
