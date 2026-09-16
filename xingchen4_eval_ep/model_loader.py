"""Load a fused or ModuleList MoE verifier with expert parallelism only."""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

from ep_plan import ExpertParallelPlan, resolve_ep_plan, validate_ep_plan, write_ep_plan
from modulelist_ep import detect_expert_layout, install_modulelist_expert_parallel
from parallel_state import ExpertParallelContext


LOGGER = logging.getLogger("dspark_pytorch_ep_eval")
MODULE_HOOK_EP_STYLES = frozenset({"ep_router", "moe_tp_experts"})


def _resolve_dtype(name: str):
    import torch

    if name == "auto":
        return "auto"
    dtype = getattr(torch, name, None)
    if dtype is None:
        raise ValueError(f"Unknown torch dtype: {name}")
    return dtype


def _available_parallel_styles() -> set[str]:
    try:
        from transformers.integrations.tensor_parallel import ALL_PARALLEL_STYLES
    except ImportError as error:
        raise RuntimeError(
            "The installed Transformers build has no native parallel integration. "
            "This evaluator requires Transformers with DistributedConfig EP support."
        ) from error
    return set(ALL_PARALLEL_STYLES)


def _distributed_config(ep_size: int):
    try:
        from transformers.distributed import DistributedConfig
    except ImportError:
        from transformers.distributed.configuration_utils import DistributedConfig

    # Transformers currently names the EP mesh width `tp_size`. With
    # enable_expert_parallel=True, model.tp_plan resolves to base_model_ep_plan.
    return DistributedConfig(
        tp_size=ep_size,
        enable_expert_parallel=True,
    )


def _parameter_parallel_style(
    parameter_name: str,
    active_plan: dict[str, str],
) -> str | None:
    """Match a parameter exactly as Transformers' plan resolver does."""
    generic_name = re.sub(
        r"\.\d+(\.|$)",
        lambda match: ".*" + match.group(1),
        parameter_name,
    )
    if generic_name in active_plan:
        return active_plan[generic_name]
    if "." in generic_name:
        parent_name = generic_name.rsplit(".", 1)[0]
        return active_plan.get(parent_name)
    return None


def _local_parameter_shape(parameter: Any) -> tuple[int, ...]:
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        DTensor = ()  # type: ignore[assignment,misc]
    if isinstance(parameter, DTensor):
        parameter = parameter.to_local()
    return tuple(parameter.shape)


def verify_applied_ep_state(
    model: Any,
    *,
    plan: ExpertParallelPlan,
    active_plan: dict[str, str],
    ep_size: int,
) -> tuple[Counter, list[tuple[str, tuple[int, ...]]]]:
    """Verify module hooks and expert-parameter sharding separately.

    ``ep_router`` and ``moe_tp_experts`` are module-hook rules. In contrast,
    ``grouped_gemm`` is applied while loading a parameter and does not have to
    appear as ``_hf_tp_plan`` on a module.
    """
    applied_style_counts = Counter(
        style
        for _name, module in model.named_modules()
        if (style := getattr(module, "_hf_tp_plan", None)) is not None
    )
    missing_hooks = MODULE_HOOK_EP_STYLES - set(applied_style_counts)
    if missing_hooks:
        raise RuntimeError(
            "Transformers selected the EP plan, but no model modules received "
            "these required EP hooks: "
            + ", ".join(sorted(missing_hooks))
            + ". Check whether base_model_ep_plan paths match Xingchen4 modules."
        )

    grouped_parameters = [
        (name, _local_parameter_shape(parameter))
        for name, parameter in model.named_parameters()
        if _parameter_parallel_style(name, active_plan) == "grouped_gemm"
    ]
    if not grouped_parameters:
        raise RuntimeError(
            "The grouped_gemm EP rules did not match any model parameters. "
            "Check gate_up_proj/down_proj names in base_model_ep_plan."
        )

    expected_local_experts = plan.num_experts // ep_size
    wrong_shapes = [
        (name, shape)
        for name, shape in grouped_parameters
        if not shape or shape[0] != expected_local_experts
    ]
    if wrong_shapes:
        examples = ", ".join(
            f"{name}={shape}" for name, shape in wrong_shapes[:4]
        )
        raise RuntimeError(
            "grouped_gemm parameters were matched but not sharded on the expert "
            f"dimension; expected dimension 0 == {expected_local_experts}, got "
            + examples
        )

    return applied_style_counts, grouped_parameters


def load_expert_parallel_target(
    *,
    model_path: Path,
    dtype: str,
    trust_remote_code: bool,
    local_files_only: bool,
    context: ExpertParallelContext,
    plan_output_path: Path,
) -> tuple[Any, ExpertParallelPlan]:
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        str(model_path),
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    plan = resolve_ep_plan(config)
    expert_layout = detect_expert_layout(model_path)
    validate_ep_plan(
        plan,
        ep_size=context.world_size,
        available_styles=(
            _available_parallel_styles() if expert_layout == "fused" else None
        ),
    )
    if context.is_primary:
        LOGGER.info("Detected verifier expert layout: %s", expert_layout)
        write_ep_plan(
            plan_output_path,
            plan,
            context.world_size,
            implementation=(
                "native_grouped_gemm"
                if expert_layout == "fused"
                else "custom_modulelist"
            ),
        )
    context.barrier()

    if expert_layout == "modulelist":
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            config=config,
            dtype=_resolve_dtype(dtype),
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            device_map={"": context.device},
        ).eval()
        patched_blocks = install_modulelist_expert_parallel(
            model,
            num_experts=plan.num_experts,
            ep_rank=context.rank,
            ep_size=context.world_size,
            process_group=context.device_mesh.get_group(),
        )
        model._dspark_ep_implementation = "custom_modulelist"
        if context.is_primary:
            LOGGER.info(
                "Installed custom ModuleList EP on %d MoE blocks; local experts=%d",
                patched_blocks,
                plan.num_experts // context.world_size,
            )
        context.barrier()
        return model, plan

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        config=config,
        dtype=_resolve_dtype(dtype),
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        distributed_config=_distributed_config(context.world_size),
        device_mesh=context.device_mesh,
    ).eval()

    distributed = getattr(model.config, "distributed_config", None)
    if distributed is None or not distributed.enable_expert_parallel:
        raise RuntimeError(
            "Transformers loaded the verifier without active expert parallelism."
        )

    active_plan = dict(getattr(model, "tp_plan", None) or {})
    active_styles = set(active_plan.values())
    non_ep_styles = active_styles - set(plan.styles)
    if non_ep_styles:
        raise RuntimeError(
            "The active model plan unexpectedly contains non-EP styles: "
            + ", ".join(sorted(non_ep_styles))
        )
    if not plan.styles.issubset(active_styles):
        missing = plan.styles - active_styles
        raise RuntimeError(
            "The active model plan is missing required EP styles: "
            + ", ".join(sorted(missing))
        )

    applied_style_counts, grouped_parameters = verify_applied_ep_state(
        model,
        plan=plan,
        active_plan=active_plan,
        ep_size=context.world_size,
    )
    if context.is_primary:
        LOGGER.info("Applied native EP module hooks: %s", dict(applied_style_counts))
        LOGGER.info(
            "Verified %d grouped_gemm expert parameters; local expert dimension=%d",
            len(grouped_parameters),
            plan.num_experts // context.world_size,
        )

    model._dspark_ep_implementation = "native_grouped_gemm"

    return model, plan


def localize_model_output(value: Any) -> Any:
    """Return an ordinary local/full tensor if an output is a DTensor."""
    try:
        from torch.distributed.tensor import DTensor, Replicate
    except ImportError:
        return value
    if not isinstance(value, DTensor):
        return value
    if all(isinstance(placement, Replicate) for placement in value.placements):
        return value.to_local()
    return value.full_tensor()


class LocalOutputTargetAdapter:
    """Expose DSpark-compatible verifier outputs.

    Xingchen4 keeps ``hc_mult`` residual streams inside every decoder layer, so
    Transformers records intermediate hidden states as ``[B, S, hc_mult, H]``.
    DSpark is trained with the model's learned hyper-head collapse and expects
    ordinary ``[B, S, H]`` features.  Apply that same learned collapse at the
    output boundary instead of silently averaging or selecting one stream.
    """

    def __init__(
        self,
        model: Any,
        target_layer_ids: list[int],
        *,
        expected_feature_width: int | None = None,
    ) -> None:
        self.model = model
        self.target_layer_ids = set(target_layer_ids)
        self.expected_feature_width = expected_feature_width
        self.generation_config = model.generation_config
        self.hidden_stream_collapser, self.hidden_stream_collapser_name = (
            _find_hidden_stream_collapser(model)
        )

    def _run_target(self, *args: Any, **kwargs: Any) -> Any:
        if self.hidden_stream_collapser is not None:
            return self.model(*args, **kwargs)

        # Private Xingchen variants do not consistently name the final mHC
        # stream head. Observe only the first target forward and identify the
        # learned module by its unambiguous rank-4 -> rank-3 shape transition.
        probe = _HiddenStreamCollapserProbe(self.model)
        try:
            output = self.model(*args, **kwargs)
        finally:
            probe.remove()
        discovered = probe.resolve()
        if discovered is not None:
            self.hidden_stream_collapser_name, self.hidden_stream_collapser = (
                discovered
            )
            LOGGER.info(
                "Discovered Xingchen mHC stream collapser by runtime shape: %s",
                self.hidden_stream_collapser_name,
            )
        return output

    def _canonical_hidden_state(self, value: Any, index: int) -> Any:
        import torch

        value = localize_model_output(value)
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(
                f"Verifier hidden_states[{index}] is not a tensor: "
                f"{type(value).__name__}"
            )
        if value.ndim == 4:
            if self.hidden_stream_collapser is None:
                raise RuntimeError(
                    "Verifier returned a four-dimensional mHC hidden state "
                    f"{tuple(value.shape)}, but no learned hc_head/hyper_head "
                    "module was found on the base model. Do not average or select "
                    "one stream; expose the model's learned stream-collapse module."
                )
            source = value
            collapse_output = self.hidden_stream_collapser(source)
            value = _extract_collapsed_hidden_tensor(
                collapse_output,
                source=source,
                module_name=self.hidden_stream_collapser_name or "<unknown>",
            )
        if value.ndim != 3:
            raise RuntimeError(
                "DSpark verifier hidden states must be [batch, sequence, hidden] "
                f"after stream collapse; hidden_states[{index}] is "
                f"{tuple(value.shape)}"
            )
        return value

    def _validate_selected_hidden_states(self, hidden_states: list[Any]) -> None:
        if self.expected_feature_width is None:
            return
        selected = [hidden_states[index] for index in sorted(self.target_layer_ids)]
        actual_width = sum(int(value.shape[-1]) for value in selected)
        if actual_width != self.expected_feature_width:
            shapes = [tuple(value.shape) for value in selected]
            raise RuntimeError(
                "Verifier/draft hidden-state contract mismatch after mHC collapse: "
                f"selected shapes={shapes}, concatenated width={actual_width}, "
                f"but draft.fc.in_features={self.expected_feature_width}. Check "
                "that the draft checkpoint's target_layer_ids match the layer IDs "
                "used to generate its Arrow training data."
            )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        output = self._run_target(*args, **kwargs)
        if hasattr(output, "logits"):
            output.logits = localize_model_output(output.logits)
        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states is not None:
            localized = list(hidden_states)
            for index in self.target_layer_ids:
                localized[index] = self._canonical_hidden_state(
                    localized[index],
                    index,
                )
            self._validate_selected_hidden_states(localized)
            output.hidden_states = tuple(localized)
        return output

    def parameters(self, *args: Any, **kwargs: Any):
        return self.model.parameters(*args, **kwargs)

    def eval(self) -> "LocalOutputTargetAdapter":
        self.model.eval()
        return self


def _find_hidden_stream_collapser(model: Any) -> tuple[Any | None, str | None]:
    """Find Xingchen/DeepSeek-V4's learned final mHC stream collapse."""
    preferred_suffixes = (
        "hc_head",
        "hyper_head",
        "hyper_connection_head",
    )
    candidates = []
    for module_name, module in model.named_modules():
        leaf_name = module_name.rsplit(".", 1)[-1]
        normalized_name = leaf_name.replace("_", "").lower()
        normalized_class = type(module).__name__.replace("_", "").lower()
        named_like_head = "head" in normalized_name and (
            "hc" in normalized_name or "hyper" in normalized_name
        )
        classed_like_head = "head" in normalized_class and (
            "hc" in normalized_class or "hyper" in normalized_class
        )
        has_hyper_head_parameters = hasattr(module, "hc_fn") and hasattr(
            module,
            "hc_base",
        )
        if callable(module) and (
            leaf_name in preferred_suffixes
            or named_like_head
            or classed_like_head
            or has_hyper_head_parameters
        ):
            candidates.append((module_name, module))
    if not candidates:
        return None, None
    def priority(item: tuple[str, Any]) -> tuple[int, int]:
        leaf_name = item[0].rsplit(".", 1)[-1]
        exact_priority = (
            preferred_suffixes.index(leaf_name)
            if leaf_name in preferred_suffixes
            else len(preferred_suffixes)
        )
        return exact_priority, item[0].count(".")

    candidates.sort(key=priority)
    name, module = candidates[0]
    return module, name


def _extract_collapsed_hidden_tensor(
    output: Any,
    *,
    source: Any,
    module_name: str,
) -> Any:
    """Extract ``[B, S, H]`` from tensor or mHC tuple-style output.

    Some Xingchen variants reuse the regular HyperConnection module for the
    final stream head. Its forward returns ``(post, comb, collapsed)`` instead
    of returning ``collapsed`` directly.
    """
    import torch

    candidates = []

    def visit(value: Any) -> None:
        value = localize_model_output(value)
        if isinstance(value, torch.Tensor):
            if (
                value.ndim == 3
                and value.shape[:2] == source.shape[:2]
                and value.shape[-1] == source.shape[-1]
            ):
                candidates.append(value)
            return
        if isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
            return
        if isinstance(value, dict):
            for item in value.values():
                visit(item)

    visit(output)
    if len(candidates) == 1:
        return candidates[0]
    output_type = type(output).__name__
    if not candidates:
        raise RuntimeError(
            f"mHC stream collapser {module_name!r} returned {output_type}, but "
            "it contained no [batch, sequence, hidden] tensor matching input "
            f"shape {tuple(source.shape)}."
        )
    raise RuntimeError(
        f"mHC stream collapser {module_name!r} returned {output_type} with "
        f"{len(candidates)} matching [batch, sequence, hidden] tensors; cannot "
        "choose the training-compatible collapsed state unambiguously."
    )


class _HiddenStreamCollapserProbe:
    """Discover a private model's learned mHC head during one real forward."""

    def __init__(self, model: Any) -> None:
        self.candidates: list[tuple[str, Any]] = []
        self.handles = []
        for module_name, module in model.named_modules():
            if not module_name:
                continue
            self.handles.append(
                module.register_forward_hook(self._make_hook(module_name))
            )

    def _make_hook(self, module_name: str):
        def observe(module: Any, inputs: Any, output: Any) -> None:
            import torch

            if not isinstance(output, torch.Tensor) or output.ndim != 3:
                return
            rank_four_inputs = [
                value
                for value in inputs
                if isinstance(value, torch.Tensor) and value.ndim == 4
            ]
            if not rank_four_inputs:
                return
            source = rank_four_inputs[0]
            if (
                source.shape[:2] == output.shape[:2]
                and source.shape[-1] == output.shape[-1]
            ):
                self.candidates.append((module_name, module))

        return observe

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def resolve(self) -> tuple[str, Any] | None:
        unique: dict[int, tuple[str, Any]] = {
            id(module): (name, module) for name, module in self.candidates
        }
        if not unique:
            return None
        candidates = list(unique.values())
        candidates.sort(
            key=lambda item: (
                0
                if "head" in item[0].lower()
                or "head" in type(item[1]).__name__.lower()
                else 1,
                item[0].count("."),
            )
        )
        return candidates[0]
