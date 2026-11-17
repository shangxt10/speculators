"""MLA math, compressed cache, opt-in routing and checkpoint regressions."""

from unittest.mock import patch

import pytest
import torch
from transformers import DynamicCache, LlamaConfig, Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from speculators import SpeculatorsConfig, VerifierConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash.model_definitions import Qwen3DFlashAttention
from speculators.models.draft_config import transformer_config_from_dict
from speculators.models.dspark.config import DSparkSpeculatorConfig
from speculators.models.dspark.core import DSparkDraftModel
from speculators.models.eagle3 import Eagle3DraftModel, Eagle3SpeculatorConfig
from speculators.models.metrics import resolve_loss_config
from speculators.models.mla import (
    Qwen3MLAAttention,
    configure_mla,
    mla_rotary_config,
)
from speculators.models.peagle.core import PEagleDraftModel
from speculators.proposals.greedy import GreedyTokenProposalConfig


def _config(**overrides):
    values = {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "use_mla": True,
        "mla_kv_lora_rank": 4,
        "mla_qk_nope_head_dim": 4,
        "mla_qk_rope_head_dim": 4,
        "mla_v_head_dim": 6,
        "attn_implementation": "sdpa",
        "eos_token_id": 31,
    }
    values.update(overrides)
    return Qwen3Config(**values)


def _rope(config, x, positions):
    return Qwen3RotaryEmbedding(mla_rotary_config(config))(x, positions)


@pytest.mark.parametrize("backend", ["eager", "sdpa", "simple_flex_attention"])
@pytest.mark.parametrize("query_rank", [None, 3])
def test_mla_matches_explicit_attention_and_backpropagates(backend, query_rank):
    from torch.nn.attention.flex_attention import create_block_mask  # noqa: PLC0415

    torch.manual_seed(10)
    config = _config(attn_implementation=backend, mla_q_lora_rank=query_rank)
    attn = Qwen3MLAAttention(config, 0)
    x = torch.randn(2, 3, 16, requires_grad=True)
    context = torch.randn(2, 4, 16, requires_grad=True)
    positions = torch.tensor([[0, 1, 2, 3, 4, 5, 6]]).expand(2, -1)
    embeddings = _rope(config, x, positions)
    mask = torch.ones(2, 1, 3, 7, dtype=torch.bool)
    mask[..., 1] = False
    if backend == "simple_flex_attention":

        def mask_mod(b, h, q, k):
            return k != 1

        actual_mask = create_block_mask(
            mask_mod, B=2, H=None, Q_LEN=3, KV_LEN=7, device="cpu"
        )
    else:
        actual_mask = mask
    supports_backward = backend != "simple_flex_attention"
    with torch.set_grad_enabled(supports_backward):
        actual, _ = attn(x, embeddings, actual_mask, target_hidden=context)

    # Independent dense reference, including the shared position key.
    q = (
        attn.q_proj(x)
        if query_rank is None
        else attn.q_b_proj(attn.q_a_layernorm(attn.q_a_proj(x)))
    )
    q = q.reshape(2, 3, 2, 8).transpose(1, 2)
    compressed = attn.kv_a_proj_with_mqa(torch.cat([context, x], dim=1))
    latent = attn.kv_a_layernorm(compressed[..., :4])
    kv = attn.kv_b_proj(latent).reshape(2, 7, 2, 10).transpose(1, 2)
    cos, sin = embeddings
    q_pos, k_pos = q[..., 4:], compressed[..., 4:].unsqueeze(1)
    q_pos = (
        q_pos * cos[:, None, -3:]
        + torch.cat([-q_pos[..., 2:], q_pos[..., :2]], -1) * sin[:, None, -3:]
    )
    k_pos = (
        k_pos * cos[:, None]
        + torch.cat([-k_pos[..., 2:], k_pos[..., :2]], -1) * sin[:, None]
    )
    query = torch.cat([q[..., :4], q_pos], dim=-1)
    key = torch.cat([kv[..., :4], k_pos.expand(-1, 2, -1, -1)], dim=-1)
    scores = (query @ key.transpose(-1, -2)) / (8**0.5)
    probs = scores.masked_fill(~mask, float("-inf")).softmax(-1)
    expected = attn.o_proj((probs @ kv[..., 4:]).transpose(1, 2).reshape(2, 3, 12))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    if not supports_backward:
        return  # PyTorch's CPU FlexAttention implements forward only.
    actual.square().mean().backward()
    assert x.grad.abs().sum() > 0
    assert context.grad.abs().sum() > 0
    for name, parameter in attn.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


@pytest.mark.parametrize("backend", ["sdpa", "eager"])
def test_cached_decoding_matches_full_sequence_and_stores_latents(backend):
    torch.manual_seed(11)
    config = _config(attn_implementation=backend, mla_kv_lora_rank=5)
    attn = Qwen3MLAAttention(config, 0).eval()
    x = torch.randn(2, 5, 16)
    positions = torch.arange(5).unsqueeze(0).expand(2, -1)
    mask = torch.ones(5, 5, dtype=torch.bool).tril()[None, None]
    full, _ = attn(x, _rope(config, x, positions), mask)
    cache = DynamicCache()
    pieces = []
    for start, end in [(0, 3), (3, 5)]:
        piece, _ = attn(
            x[:, start:end],
            _rope(config, x, positions[:, start:end]),
            mask[..., start:end, :end],
            past_key_values=cache,
        )
        pieces.append(piece)
    torch.testing.assert_close(torch.cat(pieces, dim=1), full)
    assert cache.get_seq_length() == 5
    assert cache.layers[0].keys.shape == (2, 1, 5, 4)
    assert cache.layers[0].values.shape == (2, 1, 5, 5)
    cache.crop(3)
    assert cache.get_seq_length() == 3


def _model(use_mla=True, family="dspark", layer_types=None):
    tl_config = _config(use_mla=use_mla)
    if layer_types is not None:
        values = tl_config.to_dict()
        values.pop("use_mla")
        values["layer_types"] = layer_types
        tl_config = transformer_config_from_dict(values)
        tl_config._attn_implementation = "sdpa"
    spec_config = SpeculatorsConfig(
        algorithm=family,
        proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
        default_proposal_method="greedy",
        verifier=VerifierConfig(name_or_path=None, architectures=["Qwen3ForCausalLM"]),
    )
    if family == "eagle3":
        model = Eagle3DraftModel(
            Eagle3SpeculatorConfig(
                transformer_layer_config=tl_config,
                draft_vocab_size=32,
                eagle_aux_hidden_state_layer_ids=[0],
                speculators_config=spec_config,
            )
        )
    else:
        model = DSparkDraftModel(
            DSparkSpeculatorConfig(
                transformer_layer_config=tl_config,
                draft_vocab_size=32,
                block_size=3,
                aux_hidden_state_layer_ids=[0],
                mask_token_id=0,
                markov_rank=4,
                enable_confidence_head=False,
                speculators_config=spec_config,
            )
        )
    # Frozen verifier embeddings/heads are NaN placeholders until weights load.
    with torch.no_grad():
        for parameter in model.parameters():
            if not torch.isfinite(parameter).all():
                parameter.normal_(std=0.1)
    return model


def test_legacy_attention_is_unchanged_without_opt_in():
    config = _config()
    del config.use_mla
    original = config.to_dict()
    configure_mla(config)
    assert config.to_dict() == original
    model = _model(use_mla=False)
    assert all(
        isinstance(layer.self_attn, Qwen3DFlashAttention) for layer in model.layers
    )
    assert "layers.0.self_attn.k_proj.weight" in model.state_dict()
    assert not any("kv_a_proj" in name for name in model.state_dict())


@pytest.mark.parametrize("family", ["dspark", "eagle3"])
@pytest.mark.parametrize(
    "layer_types",
    [
        None,
        ["mla", "mla"],
        ["mla", "full_attention"],
        ["full_attention", "mla"],
    ],
)
def test_mla_checkpoint_roundtrip_and_decoder_forward(tmp_path, family, layer_types):
    torch.manual_seed(12)
    model = _model(family=family, layer_types=layer_types).eval()
    expected_mla = (
        [True, True] if layer_types is None else [kind == "mla" for kind in layer_types]
    )
    assert [
        isinstance(layer.self_attn, Qwen3MLAAttention) for layer in model.layers
    ] == expected_mla
    model.save_pretrained(tmp_path)
    config = model.config_class.from_pretrained(tmp_path)
    if layer_types is None:
        assert config.transformer_layer_config.use_mla is True
    else:
        assert config.transformer_layer_config.layer_types == layer_types
        assert not getattr(config.transformer_layer_config, "use_mla", False)
    assert config.transformer_layer_config.mla_kv_lora_rank == 4
    config.transformer_layer_config._attn_implementation = "sdpa"
    with patch.object(type(model), "load_verifier_weights"):
        loaded = type(model).from_pretrained(tmp_path, config=config).eval()
    x = torch.randn(1, 3, 16 if family == "dspark" else 32)
    positions = torch.arange(5 if family == "dspark" else 3)[None]
    kwargs = {"target_hidden": torch.randn(1, 2, 16)} if family == "dspark" else {}

    def run(instance):
        hidden = x
        embeddings = instance.rotary_emb(x, positions)
        for layer in instance.layers:
            hidden = layer(
                hidden_states=hidden,
                position_embeddings=embeddings,
                position_ids=positions,
                **kwargs,
            )
        return hidden

    torch.testing.assert_close(run(model), run(loaded))


def test_dspark_mla_training_loss_and_gradients():
    torch.manual_seed(13)
    model = _model().train()
    batch = {
        "hidden_states": torch.randn(1, 16, 16),
        "input_ids": torch.randint(1, 30, (1, 16)),
        "loss_mask": torch.ones(1, 16),
        "verifier_last_hidden_states": torch.randn(1, 16, 16),
        "document_ids": torch.tensor([[0] * 8 + [1] * 8]),
    }
    _, loss, _ = model(
        **batch,
        max_anchors=3,
        loss_config=resolve_loss_config("ce"),
    )
    assert torch.isfinite(loss)
    loss.backward()
    for layer in model.layers:
        grad = layer.self_attn.kv_a_proj_with_mqa.weight.grad
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"mla_kv_lora_rank": 0}, "positive"),
        ({"mla_q_lora_rank": -1}, "positive"),
        ({"mla_qk_nope_head_dim": 0}, "positive"),
        ({"mla_qk_rope_head_dim": 3}, "even"),
        ({"mla_v_head_dim": 1.5}, "positive"),
    ],
)
def test_invalid_mla_dimensions(overrides, message):
    with pytest.raises(ValueError, match=message):
        configure_mla(_config(**overrides))


def test_mla_rejects_non_qwen_draft():
    with pytest.raises(ValueError, match="qwen3"):
        configure_mla(LlamaConfig(), draft_use_mla=True)


def test_mixed_mla_rotary_matches_explicit_mla_frequencies():
    config = _config()
    values = config.to_dict()
    values.pop("use_mla")
    values["layer_types"] = ["mla", "full_attention"]
    mixed_config = transformer_config_from_dict(values)
    mixed_config._attn_implementation = "sdpa"
    reference = Qwen3MLAAttention(config, 0)
    mixed = Qwen3MLAAttention(mixed_config, 0)
    mixed.load_state_dict(reference.state_dict())
    hidden = torch.randn(1, 3, 16)
    positions = torch.tensor([[7, 10, 25]])
    expected, _ = reference(hidden, _rope(config, hidden, positions))
    actual, _ = mixed(
        hidden,
        Qwen3RotaryEmbedding(config)(hidden, positions),
        position_ids=positions,
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "model_class",
    [
        DSparkDraftModel,
        DFlashDraftModel,
        Eagle3DraftModel,
        PEagleDraftModel,
    ],
)
@pytest.mark.parametrize("use_mla", [False, True])
def test_training_factory_routes_mla_and_full_forward(model_class, use_mla):
    config = _config(
        use_mla=False,
        use_sliding_window=True,
        sliding_window=4,
        layer_types=["sliding_attention", "full_attention"],
    )
    verifier = VerifierConfig(name_or_path=None, architectures=["Qwen3ForCausalLM"])
    with (
        patch.object(model_class, "load_verifier_weights"),
        patch.object(VerifierConfig, "from_pretrained", return_value=verifier),
    ):
        model = model_class.from_training_args(
            verifier_config=config,
            verifier_name_or_path="dummy",
            target_layer_ids=[0],
            draft_vocab_size=32,
            draft_use_mla=use_mla,
            mla_kv_lora_rank=5 if use_mla else None,
            mla_q_lora_rank=3 if use_mla else None,
            draft_attn_impl="sdpa",
            block_size=3,
            mask_token_id=0,
            markov_rank=4,
            enable_confidence_head=False,
            norm_before_residual=False,
            ttt_steps=3,
        )
    assert all(
        isinstance(layer.self_attn, Qwen3MLAAttention) == use_mla
        for layer in model.layers
    )
    with torch.no_grad():
        for parameter in model.parameters():
            if not torch.isfinite(parameter).all():
                parameter.normal_(std=0.1)
    model.train()
    _, loss, _ = model(
        hidden_states=torch.randn(1, 16, 16),
        input_ids=torch.randint(1, 30, (1, 16)),
        loss_mask=torch.ones(1, 16),
        verifier_last_hidden_states=torch.randn(1, 16, 16),
        document_ids=torch.zeros(1, 16, dtype=torch.long),
        max_anchors=3,
        num_depths=3,
        ttt_steps=3,
        loss_config=resolve_loss_config("ce"),
    )
    assert torch.isfinite(loss)
    loss.backward()
    if use_mla:
        for layer in model.layers:
            grad = layer.self_attn.kv_a_proj_with_mqa.weight.grad
            assert grad is not None
            assert torch.isfinite(grad).all()
            assert grad.abs().sum() > 0
