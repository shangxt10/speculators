"""Ascend tensor-format compatibility helpers for the replicated draft model."""

from __future__ import annotations

from collections.abc import Callable
from types import MethodType
from typing import Any


ACL_FORMAT_ND = 2


def _cast_tensor_to_nd(value: Any) -> Any:
    import torch

    if not isinstance(value, torch.Tensor) or value.device.type != "npu":
        return value

    import torch_npu

    return torch_npu.npu_format_cast(value, ACL_FORMAT_ND)


def install_draft_kv_projection_nd_hooks(
    draft_model: Any,
    *,
    cast_fn: Callable[[Any], Any] | None = None,
) -> int:
    """Normalize every draft K/V projection output before attention cat()."""
    existing = getattr(draft_model, "_dspark_npu_format_hook_handles", None)
    if existing is not None:
        return len(existing)

    formatter = cast_fn or _cast_tensor_to_nd

    def normalize_output(_module: Any, _inputs: Any, output: Any) -> Any:
        return formatter(output)

    handles = []
    for module_name, module in draft_model.named_modules():
        if module_name.endswith(("self_attn.k_proj", "self_attn.v_proj")):
            handles.append(module.register_forward_hook(normalize_output))

    if not handles:
        raise RuntimeError(
            "Could not find draft self_attn.k_proj/v_proj modules for Ascend "
            "format normalization."
        )

    draft_model._dspark_npu_format_hook_handles = tuple(handles)
    return len(handles)


def _empty_nd_like_sequence(left: Any, total_length: int) -> Any:
    """Allocate [batch, total_length, hidden] in an explicit base ND format."""
    import torch

    shape = (left.shape[0], total_length, left.shape[2])
    if left.device.type != "npu":
        return torch.empty(shape, dtype=left.dtype, device=left.device)

    import torch_npu

    empty_with_format = getattr(torch_npu, "empty_with_format", None)
    if callable(empty_with_format):
        return empty_with_format(
            shape,
            dtype=left.dtype,
            device=left.device,
            acl_format=ACL_FORMAT_ND,
        )
    return torch.empty(shape, dtype=left.dtype, device=left.device)


def _sequence_concat_without_cat(left: Any, right: Any) -> Any:
    """Concatenate sequence tensors through copies into one known-format buffer."""
    if left.ndim != 3 or right.ndim != 3:
        raise RuntimeError(
            "Draft attention K/V tensors must be rank 3, got "
            f"{tuple(left.shape)} and {tuple(right.shape)}"
        )
    if left.shape[0] != right.shape[0] or left.shape[2] != right.shape[2]:
        raise RuntimeError(
            "Draft attention K/V tensors disagree outside the sequence axis: "
            f"{tuple(left.shape)} and {tuple(right.shape)}"
        )

    left_length = left.shape[1]
    output = _empty_nd_like_sequence(left, left_length + right.shape[1])
    output.narrow(1, 0, left_length).copy_(left)
    output.narrow(1, left_length, right.shape[1]).copy_(right)
    return output


def _npu_safe_dflash_attention_forward(
    self: Any,
    hidden_states: Any,
    target_hidden: Any,
    position_embeddings: Any,
    attention_mask: Any,
    past_key_values: Any = None,
    cache_position: Any = None,
    **kwargs: Any,
):
    """Equivalent DFlash attention forward with K/V cat replaced by copies."""
    from speculators.models.dflash import model_definitions as definitions

    batch_size, query_length = hidden_states.shape[:-1]
    context_length = target_hidden.shape[1]

    query = self.q_proj(hidden_states)
    query = query.view(batch_size, query_length, -1, self.head_dim)
    query = self.q_norm(query).transpose(1, 2)

    key_context = self.k_proj(target_hidden)
    key_noise = self.k_proj(hidden_states)
    value_context = self.v_proj(target_hidden)
    value_noise = self.v_proj(hidden_states)

    key = _sequence_concat_without_cat(key_context, key_noise).view(
        batch_size,
        context_length + query_length,
        -1,
        self.head_dim,
    )
    value = _sequence_concat_without_cat(value_context, value_noise).view(
        batch_size,
        context_length + query_length,
        -1,
        self.head_dim,
    )

    key = self.k_norm(key).transpose(1, 2)
    value = value.transpose(1, 2)
    cos, sin = position_embeddings
    query, key = definitions.apply_rotary_pos_emb(query, key, cos, sin)
    if past_key_values is not None:
        cache_kwargs = {
            "sin": sin,
            "cos": cos,
            "cache_position": cache_position,
        }
        key, value = past_key_values.update(
            key,
            value,
            self.layer_idx,
            cache_kwargs,
        )

    attention_function = definitions.eager_attention_forward
    implementation = self.config._attn_implementation  # noqa: SLF001
    if implementation is not None and implementation != "eager":
        attention_function = definitions.ALL_ATTENTION_FUNCTIONS[implementation]

    attention_output, attention_weights = attention_function(
        self,
        query,
        key,
        value,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )
    attention_output = attention_output.reshape(
        batch_size,
        query_length,
        -1,
    )
    attention_output = self.o_proj(attention_output)
    return attention_output, attention_weights


def install_draft_attention_cat_compatibility(draft_model: Any) -> int:
    """Patch only DFlash attention instances; installed package files stay intact."""
    existing = getattr(draft_model, "_dspark_npu_attention_patch_count", None)
    if existing is not None:
        return existing

    patched = 0
    for _module_name, module in draft_model.named_modules():
        if type(module).__name__ != "Qwen3DFlashAttention":
            continue
        module.forward = MethodType(_npu_safe_dflash_attention_forward, module)
        patched += 1

    if patched == 0:
        raise RuntimeError(
            "Could not find Qwen3DFlashAttention modules for NPU-safe K/V "
            "concatenation."
        )

    draft_model._dspark_npu_attention_patch_count = patched
    return patched
