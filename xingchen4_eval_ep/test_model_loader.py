"""CPU-only tests for verifier output adaptation."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

if importlib.util.find_spec("torch") is not None:
    import torch
    from torch import nn

    import model_loader
    from model_loader import (
        LocalOutputTargetAdapter,
        _require_accelerate_for_low_cpu_load,
    )


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is unavailable")
class LocalOutputTargetAdapterTests(unittest.TestCase):
    def test_missing_accelerate_reports_install_command(self) -> None:
        with (
            mock.patch.dict(sys.modules, {"accelerate": None}),
            self.assertRaisesRegex(
                RuntimeError,
                r"python.* -m pip install 'accelerate>=1\.1\.0'",
            ),
        ):
            _require_accelerate_for_low_cpu_load()

    def test_modulelist_checkpoint_is_pruned_on_cpu_before_npu_transfer(self) -> None:
        events = []

        class Model:
            def eval(self):
                events.append("eval")
                return self

            def to(self, device):
                events.append(("to", device))
                return self

        class DeviceMesh:
            @staticmethod
            def get_group():
                return "ep_group"

        context = SimpleNamespace(
            is_primary=False,
            world_size=2,
            rank=0,
            device="npu:0",
            device_mesh=DeviceMesh(),
            barrier=lambda: events.append("barrier"),
        )
        plan = SimpleNamespace(num_experts=4)
        captured_kwargs = {}

        def load_model(*args, **kwargs):
            del args
            captured_kwargs.update(kwargs)
            events.append("load")
            return Model()

        def install(model, **kwargs):
            del model, kwargs
            events.append("prune")
            return 1

        with (
            mock.patch("transformers.AutoConfig.from_pretrained", return_value=object()),
            mock.patch(
                "transformers.AutoModelForCausalLM.from_pretrained",
                side_effect=load_model,
            ),
            mock.patch.object(model_loader, "resolve_ep_plan", return_value=plan),
            mock.patch.object(model_loader, "detect_expert_layout", return_value="modulelist"),
            mock.patch.object(model_loader, "validate_ep_plan"),
            mock.patch.object(
                model_loader,
                "_require_accelerate_for_low_cpu_load",
            ),
            mock.patch.object(
                model_loader,
                "install_modulelist_expert_parallel",
                side_effect=install,
            ),
        ):
            loaded, loaded_plan = model_loader.load_expert_parallel_target(
                model_path=Path("checkpoint"),
                dtype="bfloat16",
                trust_remote_code=True,
                local_files_only=True,
                context=context,
                plan_output_path=Path("plan.json"),
            )

        self.assertIs(loaded_plan, plan)
        self.assertIsInstance(loaded, Model)
        self.assertTrue(captured_kwargs["low_cpu_mem_usage"])
        self.assertNotIn("device_map", captured_kwargs)
        self.assertLess(events.index("load"), events.index("prune"))
        self.assertLess(events.index("prune"), events.index(("to", "npu:0")))

    def test_collapses_xingchen_mhc_hidden_states(self) -> None:
        class HyperHead(nn.Module):
            def forward(self, value):
                return value.sum(dim=2)

        class Base(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.hc_head = HyperHead()

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = Base()
                self.anchor = nn.Parameter(torch.zeros(()))
                self.generation_config = SimpleNamespace()

            def forward(self, *args, **kwargs):
                del args, kwargs
                packed = torch.ones(1, 3, 4, 8)
                return SimpleNamespace(
                    logits=torch.zeros(1, 3, 11),
                    hidden_states=(packed,),
                )

        adapter = LocalOutputTargetAdapter(
            Model(),
            [0],
            expected_feature_width=8,
        )
        output = adapter()

        self.assertEqual(tuple(output.hidden_states[0].shape), (1, 3, 8))
        self.assertTrue(torch.equal(output.hidden_states[0], torch.full((1, 3, 8), 4.0)))
        self.assertEqual(adapter.hidden_stream_collapser_name, "model.hc_head")

    def test_discovers_privately_named_stream_collapser_by_shape(self) -> None:
        class StreamMerger(nn.Module):
            def forward(self, value):
                return value.sum(dim=2)

        class Base(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.stream_merger = StreamMerger()

            def forward(self, packed):
                return self.stream_merger(packed)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = Base()
                self.anchor = nn.Parameter(torch.zeros(()))
                self.generation_config = SimpleNamespace()

            def forward(self, *args, **kwargs):
                del args, kwargs
                packed = torch.ones(1, 3, 4, 8)
                collapsed = self.model(packed)
                return SimpleNamespace(
                    logits=torch.zeros(1, 3, 11) + collapsed[..., :1],
                    hidden_states=(packed,),
                )

        adapter = LocalOutputTargetAdapter(
            Model(),
            [0],
            expected_feature_width=8,
        )
        output = adapter()

        self.assertEqual(tuple(output.hidden_states[0].shape), (1, 3, 8))
        self.assertEqual(
            adapter.hidden_stream_collapser_name,
            "model.stream_merger",
        )

    def test_extracts_collapsed_state_from_mhc_tuple(self) -> None:
        class TupleHyperHead(nn.Module):
            def forward(self, value):
                batch, sequence, streams, _hidden = value.shape
                post = value.new_ones(batch, sequence, streams)
                comb = value.new_zeros(batch, sequence, streams, streams)
                collapsed = value.sum(dim=2)
                return post, comb, collapsed

        class Base(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.output_hc_head = TupleHyperHead()

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = Base()
                self.anchor = nn.Parameter(torch.zeros(()))
                self.generation_config = SimpleNamespace()

            def forward(self, *args, **kwargs):
                del args, kwargs
                packed = torch.ones(1, 3, 4, 8)
                return SimpleNamespace(
                    logits=torch.zeros(1, 3, 11),
                    hidden_states=(packed,),
                )

        adapter = LocalOutputTargetAdapter(
            Model(),
            [0],
            expected_feature_width=8,
        )
        output = adapter()

        self.assertEqual(tuple(output.hidden_states[0].shape), (1, 3, 8))
        self.assertTrue(
            torch.equal(output.hidden_states[0], torch.full((1, 3, 8), 4.0))
        )

    def test_rejects_wrong_draft_feature_width(self) -> None:
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(()))
                self.generation_config = SimpleNamespace()

            def forward(self, *args, **kwargs):
                del args, kwargs
                return SimpleNamespace(
                    logits=torch.zeros(1, 2, 11),
                    hidden_states=(torch.zeros(1, 2, 8),),
                )

        adapter = LocalOutputTargetAdapter(
            Model(),
            [0],
            expected_feature_width=16,
        )

        with self.assertRaisesRegex(RuntimeError, "draft.fc.in_features=16"):
            adapter()


if __name__ == "__main__":
    unittest.main()
