"""Pure EP backend for Xingchen-style per-expert ``ModuleList`` models."""

from __future__ import annotations

import gc
import json
import re
from pathlib import Path
from types import MethodType
from typing import Any, Literal


ExpertLayout = Literal["fused", "modulelist"]

_MODULELIST_EXPERT = re.compile(
    r"(?:^|\.)experts\.\d+\.(?:gate_proj|up_proj|down_proj)(?:\.|$)"
)
_FUSED_EXPERT = re.compile(
    r"(?:^|\.)experts\.(?:gate_up_proj|down_proj)(?:\.|$)"
)


def _checkpoint_keys(model_path: Path) -> list[str]:
    for index_name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        index_path = model_path / index_name
        if index_path.is_file():
            with index_path.open(encoding="utf-8") as file_obj:
                index = json.load(file_obj)
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict):
                raise ValueError(f"{index_path}: missing object field 'weight_map'")
            return list(weight_map)

    safetensors_path = model_path / "model.safetensors"
    if safetensors_path.is_file():
        try:
            from safetensors import safe_open
        except ImportError as error:
            raise RuntimeError(
                "Inspecting a single-file checkpoint requires safetensors."
            ) from error
        with safe_open(
            str(safetensors_path),
            framework="pt",
            device="cpu",
        ) as file_obj:
            return list(file_obj.keys())

    raise FileNotFoundError(
        "Could not inspect expert layout: expected a safetensors/bin index or "
        f"model.safetensors under {model_path}"
    )


def detect_expert_layout(model_path: Path) -> ExpertLayout:
    """Determine whether checkpoint experts are fused or individually stored."""
    keys = _checkpoint_keys(model_path)
    modulelist_keys = [key for key in keys if _MODULELIST_EXPERT.search(key)]
    fused_keys = [key for key in keys if _FUSED_EXPERT.search(key)]

    if modulelist_keys and fused_keys:
        raise ValueError(
            "Checkpoint contains both per-expert ModuleList and fused expert "
            "parameters; refusing an ambiguous EP layout."
        )
    if modulelist_keys:
        return "modulelist"
    if fused_keys:
        return "fused"

    expert_examples = [key for key in keys if ".experts." in key][:8]
    raise ValueError(
        "Could not recognize verifier expert layout. Example expert keys: "
        + repr(expert_examples)
    )


def _synchronize_ep_inputs(
    hidden_states: Any,
    topk_indices: Any,
    topk_weights: Any,
    group: Any,
) -> tuple[Any, Any, Any]:
    """Use rank 0 routing decisions and inputs on every EP rank."""
    import torch.distributed as dist

    hidden_states = hidden_states.contiguous()
    topk_indices = topk_indices.contiguous()
    topk_weights = topk_weights.contiguous()
    dist.broadcast(hidden_states, src=0, group=group)
    dist.broadcast(topk_indices, src=0, group=group)
    dist.broadcast(topk_weights, src=0, group=group)
    return hidden_states, topk_indices, topk_weights


def _modulelist_ep_moe(
    self: Any,
    hidden_states: Any,
    topk_indices: Any,
    topk_weights: Any,
):
    """Run rank-local experts and combine one fixed-shape output per MoE block."""
    import torch
    import torch.distributed as dist

    if hidden_states.ndim != 2:
        raise RuntimeError(
            "ModuleList EP expects flattened hidden_states [tokens, hidden], got "
            f"{tuple(hidden_states.shape)}"
        )
    if topk_indices.shape != topk_weights.shape:
        raise RuntimeError(
            "topk_indices and topk_weights must have identical shapes, got "
            f"{tuple(topk_indices.shape)} and {tuple(topk_weights.shape)}"
        )

    group = self._dspark_ep_group
    hidden_states, topk_indices, topk_weights = _synchronize_ep_inputs(
        hidden_states,
        topk_indices,
        topk_weights,
        group,
    )
    combined = torch.zeros_like(hidden_states)
    expert_offset = self._dspark_ep_expert_offset

    for local_index, expert in enumerate(self.experts):
        global_index = expert_offset + local_index
        token_indices, topk_slots = torch.where(topk_indices == global_index)
        if token_indices.numel() == 0:
            continue

        expert_input = hidden_states.index_select(0, token_indices)
        expert_output = expert(expert_input)
        weights = topk_weights[token_indices, topk_slots].unsqueeze(-1)
        weighted_output = expert_output * weights.to(expert_output.dtype)
        combined.index_add_(
            0,
            token_indices,
            weighted_output.to(combined.dtype),
        )

    dist.all_reduce(combined, op=dist.ReduceOp.SUM, group=group)
    return combined


def install_modulelist_expert_parallel(
    model: Any,
    *,
    num_experts: int,
    ep_rank: int,
    ep_size: int,
    process_group: Any,
) -> int:
    """Prune every sparse block to rank-local experts and replace its MoE loop."""
    import torch
    from torch import nn

    if num_experts % ep_size != 0:
        raise ValueError(
            f"num_experts={num_experts} is not divisible by ep_size={ep_size}"
        )
    local_experts = num_experts // ep_size
    start = ep_rank * local_experts
    stop = start + local_experts

    patched = 0
    for _module_name, module in list(model.named_modules()):
        experts = getattr(module, "experts", None)
        if not isinstance(experts, nn.ModuleList):
            continue
        if len(experts) != num_experts or not callable(getattr(module, "moe", None)):
            continue

        module.experts = nn.ModuleList(list(experts[start:stop]))
        module._dspark_ep_global_num_experts = num_experts
        module._dspark_ep_expert_offset = start
        module._dspark_ep_group = process_group
        module.moe = MethodType(_modulelist_ep_moe, module)
        patched += 1

    if patched == 0:
        raise RuntimeError(
            "Checkpoint keys use per-expert ModuleList layout, but no model block "
            f"contained {num_experts} experts and a callable moe() method."
        )

    gc.collect()
    empty_cache = getattr(getattr(torch, "npu", None), "empty_cache", None)
    if callable(empty_cache):
        empty_cache()
    return patched
