"""CLI opt-in and initialization-source validation for draft MLA."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.train import (
    build_draft_model,
    load_draft_transformer_layer_config,
    parse_args,
)
from speculators import VerifierConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.draft_config import transformer_config_from_dict
from speculators.models.dspark.core import DSparkDraftModel
from speculators.models.eagle3 import Eagle3DraftModel
from speculators.models.mla import Qwen3MLAAttention, configure_mla
from speculators.models.peagle.core import PEagleDraftModel
from tests.unit.models.test_mla import _config


def _parse(monkeypatch, *extra):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--verifier-name-or-path",
            "dummy",
            "--speculator-type",
            "dspark",
            *extra,
        ],
    )
    return parse_args()


def test_mla_is_opt_in_and_dimensions_reach_config(monkeypatch):
    assert _parse(monkeypatch).draft_use_mla is False
    args = _parse(
        monkeypatch,
        "--draft-use-mla",
        "--mla-kv-lora-rank",
        "7",
        "--mla-q-lora-rank",
        "5",
    )
    config = _config(use_mla=False)
    configure_mla(config, **vars(args))
    assert config.use_mla is True
    assert config.mla_kv_lora_rank == 7
    assert config.mla_q_lora_rank == 5


def test_mla_alias(monkeypatch):
    assert _parse(monkeypatch, "--use-mla").draft_use_mla is True


@pytest.mark.parametrize(
    "extra",
    [
        ["--mla-kv-lora-rank", "4"],
        ["--draft-use-mla", "--mla-kv-lora-rank", "0"],
        ["--draft-use-mla", "--mla-qk-rope-head-dim", "3"],
        ["--draft-use-mla", "--draft-arch", "llama"],
        ["--draft-use-mla", "--speculator-type", "mtp"],
        ["--draft-use-mla", "--from-pretrained", "checkpoint"],
        ["--use-mla", "--draft-config", "config.json"],
    ],
)
def test_invalid_mla_cli_fails_early(monkeypatch, extra):
    with pytest.raises(SystemExit):
        _parse(monkeypatch, *extra)


def test_user_five_layer_config_loads_without_mla_flag(monkeypatch):
    path = (
        Path(__file__).resolve().parents[3]
        / "examples/train/configs/qwen3_3584_mla.json"
    )
    monkeypatch.setattr(
        "scripts.train.get_verifier_config",
        lambda *a, **kw: SimpleNamespace(
            hidden_size=3584,
            vocab_size=131072,
        ),
    )
    config = load_draft_transformer_layer_config(str(path), "dummy")
    configure_mla(config)
    assert config.layer_types == ["mla"] * 5
    assert config.hidden_size == 3584
    assert config.num_attention_heads == 32
    assert config.mla_kv_lora_rank == 512
    assert config.mla_qk_nope_head_dim == 64
    assert config.mla_qk_rope_head_dim == 64
    assert config.mla_v_head_dim == 128


@pytest.mark.parametrize(
    "layer_types",
    [["mla", "mla"], ["full_attention", "mla"], ["mla", "sliding_attention"]],
)
@pytest.mark.parametrize(
    ("family", "model_class"),
    [
        ("dspark", DSparkDraftModel),
        ("dflash", DFlashDraftModel),
        ("eagle3", Eagle3DraftModel),
        ("peagle", PEagleDraftModel),
    ],
)
def test_draft_config_builds_selected_mla_layers(
    tmp_path, monkeypatch, layer_types, family, model_class
):
    import torch  # noqa: PLC0415

    values = _config().to_dict()
    values.pop("use_mla")
    values.update(layer_types=layer_types, use_sliding_window=True, sliding_window=4)
    path = tmp_path / "draft_config.json"
    path.write_text(json.dumps(values))
    args = _parse(
        monkeypatch,
        "--speculator-type",
        family,
        "--draft-config",
        str(path),
        "--draft-attn-impl",
        "sdpa",
        "--target-layer-ids",
        "0",
        "--mask-token-id",
        "0",
        "--block-size",
        "3",
    )
    assert args.draft_use_mla is False
    monkeypatch.setattr(
        "scripts.train.get_verifier_config",
        lambda *a, **kw: SimpleNamespace(
            hidden_size=16,
            vocab_size=32,
        ),
    )
    with (
        patch.object(model_class, "load_verifier_weights"),
        patch.object(
            VerifierConfig,
            "from_pretrained",
            return_value=VerifierConfig(
                name_or_path=None,
                architectures=["Qwen3ForCausalLM"],
            ),
        ),
    ):
        model = build_draft_model(args, model_class, None, None, 32)
    assert [
        isinstance(layer.self_attn, Qwen3MLAAttention) for layer in model.layers
    ] == [kind == "mla" for kind in layer_types]
    assert model.sliding_window_indices == [
        i for i, kind in enumerate(layer_types) if kind == "sliding_attention"
    ]
    assert model.uses_full_attn is True
    with torch.no_grad():
        for parameter in model.parameters():
            if not torch.isfinite(parameter).all():
                parameter.normal_(std=0.1)
    _, loss, _ = model(
        hidden_states=torch.randn(1, 16, 16),
        input_ids=torch.randint(1, 30, (1, 16)),
        document_ids=torch.zeros(1, 16, dtype=torch.long),
        loss_mask=torch.ones(1, 16),
        verifier_last_hidden_states=torch.randn(1, 16, 16),
        max_anchors=3,
        num_depths=3,
    )
    assert torch.isfinite(loss)
    loss.backward()
    for layer in model.layers:
        if isinstance(layer.self_attn, Qwen3MLAAttention):
            grad = layer.self_attn.kv_a_proj_with_mqa.weight.grad
            assert grad is not None
            assert torch.isfinite(grad).all()
            assert grad.abs().sum() > 0


@pytest.mark.parametrize(
    "values",
    [
        {"model_type": "llama", "num_hidden_layers": 1, "layer_types": ["mla"]},
        {"model_type": "qwen3", "num_hidden_layers": 2, "layer_types": ["mla"]},
        {"model_type": "qwen3", "num_hidden_layers": 1, "layer_types": "mla"},
    ],
)
def test_invalid_mla_layer_config_is_rejected(values):
    with pytest.raises(ValueError):
        transformer_config_from_dict(values)
