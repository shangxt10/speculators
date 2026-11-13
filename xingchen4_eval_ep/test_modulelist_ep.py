"""CPU tests for checkpoint layout detection and ModuleList EP installation."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from modulelist_ep import (  # noqa: E402
    detect_expert_layout,
    install_modulelist_expert_parallel,
)


class ModuleListExpertParallelTests(unittest.TestCase):
    def _write_index(self, directory: Path, keys: list[str]) -> None:
        weight_map = {key: "model-00001-of-00001.safetensors" for key in keys}
        with (directory / "model.safetensors.index.json").open(
            "w",
            encoding="utf-8",
        ) as file_obj:
            json.dump({"weight_map": weight_map}, file_obj)

    def test_detects_modulelist_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self._write_index(
                directory,
                [
                    "model.layers.0.mlp.experts.0.gate_proj.weight",
                    "model.layers.0.mlp.experts.0.up_proj.weight",
                    "model.layers.0.mlp.experts.0.down_proj.weight",
                ],
            )
            self.assertEqual(detect_expert_layout(directory), "modulelist")

    def test_detects_fused_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self._write_index(
                directory,
                [
                    "model.layers.0.mlp.experts.gate_up_proj",
                    "model.layers.0.mlp.experts.down_proj",
                ],
            )
            self.assertEqual(detect_expert_layout(directory), "fused")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is unavailable")
    def test_installer_keeps_only_rank_local_experts(self) -> None:
        import torch
        from torch import nn

        class Expert(nn.Module):
            def __init__(self, global_id: int) -> None:
                super().__init__()
                self.global_id = global_id

        class SparseMlp(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.experts = nn.ModuleList([Expert(index) for index in range(4)])

            def moe(self, hidden_states, topk_indices, topk_weights):
                del topk_indices, topk_weights
                return torch.zeros_like(hidden_states)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mlp = SparseMlp()

        model = Model()
        patched = install_modulelist_expert_parallel(
            model,
            num_experts=4,
            ep_rank=1,
            ep_size=2,
            process_group="ep_group",
        )

        self.assertEqual(patched, 1)
        self.assertEqual(
            [expert.global_id for expert in model.mlp.experts],
            [2, 3],
        )
        self.assertEqual(model.mlp._dspark_ep_expert_offset, 2)
        self.assertEqual(model.mlp._dspark_ep_group, "ep_group")


if __name__ == "__main__":
    unittest.main()
