"""CPU-only tests for pure expert-parallel plan validation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from ep_plan import resolve_ep_plan, validate_ep_plan  # noqa: E402
from model_loader import verify_applied_ep_state  # noqa: E402


EP_PLAN = {
    "layers.*.mlp.gate": "ep_router",
    "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
    "layers.*.mlp.experts.down_proj": "grouped_gemm",
    "layers.*.mlp.experts": "moe_tp_experts",
}


class ExpertParallelPlanTests(unittest.TestCase):
    def test_accepts_xingchen_ep4(self) -> None:
        config = SimpleNamespace(base_model_ep_plan=EP_PLAN, num_experts=64)
        plan = resolve_ep_plan(config)
        validate_ep_plan(
            plan,
            ep_size=4,
            available_styles=set(EP_PLAN.values()),
        )
        self.assertEqual(plan.num_experts // 4, 16)
        self.assertEqual(plan.config_path, "config")

    def test_resolves_nested_language_config(self) -> None:
        config = SimpleNamespace(
            base_model_ep_plan=None,
            text_config=SimpleNamespace(
                base_model_ep_plan=EP_PLAN,
                n_routed_experts=64,
            ),
        )
        plan = resolve_ep_plan(config)
        self.assertEqual(plan.config_path, "config.text_config")
        self.assertEqual(plan.num_experts, 64)

    def test_rejects_non_divisible_expert_count(self) -> None:
        config = SimpleNamespace(base_model_ep_plan=EP_PLAN, num_experts=64)
        with self.assertRaisesRegex(ValueError, "not divisible"):
            validate_ep_plan(resolve_ep_plan(config), ep_size=3)

    def test_rejects_mixed_tp_rule(self) -> None:
        mixed = dict(EP_PLAN)
        mixed["layers.*.self_attn.q_proj"] = "colwise"
        config = SimpleNamespace(base_model_ep_plan=mixed, num_experts=64)
        with self.assertRaisesRegex(ValueError, "non-EP"):
            validate_ep_plan(resolve_ep_plan(config), ep_size=4)

    def test_rejects_missing_runtime_style(self) -> None:
        config = SimpleNamespace(base_model_ep_plan=EP_PLAN, num_experts=64)
        with self.assertRaisesRegex(RuntimeError, "lacks EP styles"):
            validate_ep_plan(
                resolve_ep_plan(config),
                ep_size=4,
                available_styles={"ep_router", "moe_tp_experts"},
            )

    def test_grouped_gemm_is_validated_as_parameter_sharding(self) -> None:
        config = SimpleNamespace(base_model_ep_plan=EP_PLAN, num_experts=64)
        plan = resolve_ep_plan(config)
        modules = [
            ("model.layers.0.mlp.gate", SimpleNamespace(_hf_tp_plan="ep_router")),
            (
                "model.layers.0.mlp.experts",
                SimpleNamespace(_hf_tp_plan="moe_tp_experts"),
            ),
        ]
        parameters = [
            (
                "model.layers.0.mlp.experts.gate_up_proj",
                SimpleNamespace(shape=(16, 3584, 512)),
            ),
            (
                "model.layers.0.mlp.experts.down_proj",
                SimpleNamespace(shape=(16, 512, 3584)),
            ),
        ]
        model = SimpleNamespace(
            named_modules=lambda: modules,
            named_parameters=lambda: parameters,
        )
        active_plan = {f"model.{key}": value for key, value in EP_PLAN.items()}

        hooks, grouped = verify_applied_ep_state(
            model,
            plan=plan,
            active_plan=active_plan,
            ep_size=4,
        )

        self.assertEqual(hooks["ep_router"], 1)
        self.assertEqual(len(grouped), 2)

    def test_rejects_unsharded_grouped_gemm_parameter(self) -> None:
        config = SimpleNamespace(base_model_ep_plan=EP_PLAN, num_experts=64)
        plan = resolve_ep_plan(config)
        model = SimpleNamespace(
            named_modules=lambda: [
                ("gate", SimpleNamespace(_hf_tp_plan="ep_router")),
                ("experts", SimpleNamespace(_hf_tp_plan="moe_tp_experts")),
            ],
            named_parameters=lambda: [
                (
                    "model.layers.0.mlp.experts.gate_up_proj",
                    SimpleNamespace(shape=(64, 3584, 512)),
                )
            ],
        )
        active_plan = {f"model.{key}": value for key, value in EP_PLAN.items()}

        with self.assertRaisesRegex(RuntimeError, "not sharded"):
            verify_applied_ep_state(
                model,
                plan=plan,
                active_plan=active_plan,
                ep_size=4,
            )


if __name__ == "__main__":
    unittest.main()
