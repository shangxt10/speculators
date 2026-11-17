"""Real-model, offline-only MLA checkpoint and speculative decoding regressions."""

import logging
from types import SimpleNamespace

import pytest
import torch
from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM

from speculators.models.mla import Qwen3MLAAttention
from tests.unit.evaluate.test_dspark_offline_eval import _load_module
from tests.unit.models.test_mla import _model
from xingchen4_eval_ep.npu_format import install_draft_attention_cat_compatibility


@pytest.fixture
def checkpoint_factory(tmp_path):
    torch.manual_seed(42)
    target = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            attn_implementation="sdpa",
        )
    ).eval()
    target_path = tmp_path / "target"
    target.save_pretrained(target_path)

    def make(layout, dtype="float32", sample_from_anchor="true"):
        draft = _model(layer_types=layout).eval()
        draft_path = tmp_path / "draft"
        # Stale training path must be replaced by the evaluation target path.
        draft.config.speculators_config.verifier.name_or_path = (
            "/missing/training/target"
        )
        draft.save_pretrained(draft_path)
        args = SimpleNamespace(
            draft_model=draft_path,
            verifier_model=target_path,
            device="cpu",
            draft_attn_impl="sdpa",
            dtype=dtype,
            d2t_path=None,
            t2d_path=None,
            sample_from_anchor=sample_from_anchor,
            temperature=0.0,
            tree_top_k=1,
            tree_max_nodes=7,
            max_new_tokens=12,
        )
        return target, args

    return make


@pytest.mark.parametrize(
    "layout",
    [["mla", "mla"], ["mla", "full_attention"], ["full_attention", "full_attention"]],
)
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_shared_loader_restores_structure_weights_and_dtype(
    checkpoint_factory, layout, dtype, caplog
):
    target, args = checkpoint_factory(layout, dtype=dtype)
    module = _load_module()
    with caplog.at_level(logging.INFO):
        draft = module.load_dspark_draft(args)
    assert [
        isinstance(layer.self_attn, Qwen3MLAAttention) for layer in draft.layers
    ] == [kind == "mla" for kind in layout]
    assert all(param.dtype == getattr(torch, dtype) for param in draft.parameters())
    assert draft.config.speculators_config.verifier.name_or_path == str(
        args.verifier_model
    )
    torch.testing.assert_close(
        draft.verifier_lm_head.weight, target.lm_head.weight.to(getattr(torch, dtype))
    )
    for index, layer in enumerate(draft.layers):
        assert f"Draft layer {index}: {type(layer.self_attn).__name__}" in caplog.text
    if "mla" in layout:
        assert "kv_rank=4" in caplog.text


@pytest.mark.parametrize("layout", [["mla", "mla"], ["full_attention", "mla"]])
@pytest.mark.parametrize("mode", ["token", "tree"])
@pytest.mark.parametrize("patched", [False, True])
@pytest.mark.parametrize("sample_from_anchor", ["true", "false"])
def test_mla_decoding_matches_target_greedy(
    checkpoint_factory, layout, mode, patched, sample_from_anchor
):
    target, args = checkpoint_factory(layout, sample_from_anchor=sample_from_anchor)
    module = _load_module()
    module.DynamicCache = DynamicCache
    draft = module.load_dspark_draft(args)
    if patched:
        original_forwards = [layer.self_attn.forward for layer in draft.layers]
        assert install_draft_attention_cat_compatibility(draft) == len(layout)
        assert install_draft_attention_cat_compatibility(draft) == len(layout)
        for layer, original in zip(draft.layers, original_forwards, strict=True):
            if isinstance(layer.self_attn, Qwen3MLAAttention):
                assert layer.self_attn.forward == original
    args.tree_top_k = 2 if mode == "tree" else 1

    def tokenizer(*args, **kwargs):
        return SimpleNamespace(input_ids=torch.tensor([[1, 3, 5, 7]]))

    runner = module.DSparkOfflineRunner(target, draft, tokenizer, args)
    seen = []
    handles = [
        layer.self_attn.register_forward_hook(lambda *a: seen.append(1))
        for layer in draft.layers
        if isinstance(layer.self_attn, Qwen3MLAAttention)
    ]
    try:
        actual = runner.generate_one("prompt", stop_token_ids=None)
        expected = module.BaseModelOfflineRunner(target, tokenizer, args).generate_one(
            "prompt", stop_token_ids=None
        )
    finally:
        for handle in handles:
            handle.remove()
    assert seen, "The decoder must execute actual MLA layers"
    assert len(actual.proposal_lengths) > 1, "Exercise prefix growth and cache updates"
    assert actual.num_output_tokens == args.max_new_tokens
    torch.testing.assert_close(actual.output_ids, expected.output_ids, rtol=0, atol=0)


def test_ep_loader_accepts_all_mla_checkpoint(checkpoint_factory):
    from xingchen4_eval_ep import evaluate_ep  # noqa: PLC0415

    _, args = checkpoint_factory(["mla", "mla"])
    legacy = evaluate_ep._load_legacy_evaluator()
    context = SimpleNamespace(device=torch.device("cpu"), is_primary=True)
    draft = evaluate_ep._load_draft(legacy, args, context)
    assert all(isinstance(layer.self_attn, Qwen3MLAAttention) for layer in draft.layers)
    assert draft._dspark_npu_attention_patch_count == 2


@pytest.mark.parametrize("layout", [["mla", "mla"], ["mla", "full_attention"]])
@pytest.mark.parametrize("backend", ["sdpa", "eager"])
def test_ep_concat_patch_preserves_draft_hidden_and_logits(
    checkpoint_factory, layout, backend
):
    target, args = checkpoint_factory(layout)
    args.draft_attn_impl = backend
    module = _load_module()
    draft = module.load_dspark_draft(args)
    runner = module.DSparkOfflineRunner(target, draft, None, args)
    hidden = torch.randn(1, 4, 16)
    token_ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.inference_mode():
        before = runner._single_anchor_backbone(hidden, token_ids, 4)
        install_draft_attention_cat_compatibility(draft)
        after = runner._single_anchor_backbone(hidden, token_ids, 4)
    for expected, actual in zip(before, after, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
