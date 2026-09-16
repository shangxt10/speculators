"""Validation helpers for a Transformers native expert-parallel plan."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PURE_EP_STYLES = frozenset({"ep_router", "grouped_gemm", "moe_tp_experts"})
REQUIRED_EP_STYLES = PURE_EP_STYLES
EXPERT_COUNT_NAMES = (
    "num_experts",
    "n_routed_experts",
    "num_local_experts",
)
NESTED_CONFIG_NAMES = ("text_config", "language_config", "llm_config")


@dataclass(frozen=True)
class ExpertParallelPlan:
    rules: dict[str, str]
    num_experts: int
    config_path: str

    @property
    def styles(self) -> frozenset[str]:
        return frozenset(self.rules.values())

    def to_dict(
        self,
        ep_size: int,
        *,
        implementation: str = "native_grouped_gemm",
    ) -> dict[str, Any]:
        return {
            "parallelism": "expert_parallel_only",
            "implementation": implementation,
            "ep_size": ep_size,
            "num_experts": self.num_experts,
            "local_experts_per_rank": self.num_experts // ep_size,
            "config_path": self.config_path,
            "rules": self.rules,
        }


def _candidate_configs(config: Any):
    """Yield a top-level config and common nested language-model configs."""
    seen: set[int] = set()
    pending = [("config", config)]
    while pending:
        path, candidate = pending.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        yield path, candidate
        for name in NESTED_CONFIG_NAMES:
            child = getattr(candidate, name, None)
            if child is not None:
                pending.append((f"{path}.{name}", child))


def resolve_ep_plan(config: Any) -> ExpertParallelPlan:
    """Find the model's native EP plan and routed-expert count."""
    plan_candidate: tuple[str, dict[str, str]] | None = None
    expert_count_candidate: tuple[str, int] | None = None

    for path, candidate in _candidate_configs(config):
        raw_plan = getattr(candidate, "base_model_ep_plan", None)
        if raw_plan and plan_candidate is None:
            if not isinstance(raw_plan, dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in raw_plan.items()
            ):
                raise ValueError(
                    f"{path}.base_model_ep_plan must be a string-to-string mapping"
                )
            plan_candidate = (path, dict(raw_plan))

        for name in EXPERT_COUNT_NAMES:
            value = getattr(candidate, name, None)
            if isinstance(value, int) and value > 0:
                expert_count_candidate = (path, value)
                break

        if plan_candidate is not None and expert_count_candidate is not None:
            break

    if plan_candidate is None:
        raise ValueError(
            "The verifier config has no base_model_ep_plan. Native Transformers "
            "expert parallelism is unavailable for this checkpoint."
        )
    if expert_count_candidate is None:
        raise ValueError(
            "Could not find num_experts, n_routed_experts, or num_local_experts "
            "in the verifier config."
        )

    config_path, rules = plan_candidate
    _, num_experts = expert_count_candidate
    return ExpertParallelPlan(
        rules=rules,
        num_experts=num_experts,
        config_path=config_path,
    )


def validate_ep_plan(
    plan: ExpertParallelPlan,
    *,
    ep_size: int,
    available_styles: set[str] | None = None,
) -> None:
    """Reject invalid or mixed EP/TP plans before allocating model memory."""
    if ep_size < 2:
        raise ValueError("Pure EP evaluation requires --expert-parallel-size >= 2")
    if plan.num_experts % ep_size != 0:
        raise ValueError(
            f"num_experts={plan.num_experts} is not divisible by EP size {ep_size}"
        )

    missing_kinds = REQUIRED_EP_STYLES - plan.styles
    if missing_kinds:
        raise ValueError(
            "base_model_ep_plan is incomplete; missing styles: "
            + ", ".join(sorted(missing_kinds))
        )

    non_ep_styles = plan.styles - PURE_EP_STYLES
    if non_ep_styles:
        raise ValueError(
            "Pure EP mode refuses non-EP parallel styles: "
            + ", ".join(sorted(non_ep_styles))
        )

    if available_styles is not None:
        unavailable = plan.styles - available_styles
        if unavailable:
            raise RuntimeError(
                "The installed Transformers build lacks EP styles: "
                + ", ".join(sorted(unavailable))
            )


def write_ep_plan(
    path: Path,
    plan: ExpertParallelPlan,
    ep_size: int,
    *,
    implementation: str = "native_grouped_gemm",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(
            plan.to_dict(ep_size, implementation=implementation),
            file_obj,
            ensure_ascii=False,
            indent=2,
        )
        file_obj.write("\n")
