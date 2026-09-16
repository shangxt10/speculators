"""CPU-only tests for verifier output adaptation."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

if importlib.util.find_spec("torch") is not None:
    import torch
    from torch import nn

    from model_loader import LocalOutputTargetAdapter


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is unavailable")
class LocalOutputTargetAdapterTests(unittest.TestCase):
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
