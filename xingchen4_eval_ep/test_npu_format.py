"""CPU-only tests for draft NPU format-hook registration."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from npu_format import (  # noqa: E402
    install_draft_attention_cat_compatibility,
    install_draft_kv_projection_nd_hooks,
)


class _Projection:
    def __init__(self) -> None:
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return hook

    def emit(self, output):
        for hook in self.hooks:
            output = hook(self, (), output)
        return output


class _Draft:
    def __init__(self) -> None:
        self.projections = {
            "layers.0.self_attn.k_proj": _Projection(),
            "layers.0.self_attn.v_proj": _Projection(),
        }

    def named_modules(self):
        return self.projections.items()


class Qwen3DFlashAttention:
    def forward(self):
        return "original"


class _AttentionDraft:
    def __init__(self) -> None:
        self.attention = Qwen3DFlashAttention()

    def named_modules(self):
        return [("layers.0.self_attn", self.attention)]


class NpuFormatHookTests(unittest.TestCase):
    def test_registers_hooks_once(self) -> None:
        model = _Draft()
        seen = []

        def record(value):
            seen.append(value)
            return value + 1

        self.assertEqual(
            install_draft_kv_projection_nd_hooks(model, cast_fn=record),
            2,
        )
        self.assertEqual(
            install_draft_kv_projection_nd_hooks(model, cast_fn=record),
            2,
        )
        self.assertEqual(
            model.projections["layers.0.self_attn.k_proj"].emit(4),
            5,
        )
        self.assertEqual(seen, [4])

    def test_replaces_dflash_attention_forward_once(self) -> None:
        model = _AttentionDraft()
        original = model.attention.forward

        self.assertEqual(install_draft_attention_cat_compatibility(model), 1)
        self.assertEqual(install_draft_attention_cat_compatibility(model), 1)
        self.assertNotEqual(model.attention.forward, original)


if __name__ == "__main__":
    unittest.main()
