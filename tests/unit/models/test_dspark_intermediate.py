"""Training gradients, aligned masks, scheduling and inference invariance."""

import copy
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen3Config

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.dspark.config import DSparkSpeculatorConfig
from speculators.models.dspark.core import DSparkDraftModel
from speculators.models.dspark.intermediate import (
    IntermediateHead,
    IntermediateSupervisionConfig,
    auxiliary_loss,
)
from speculators.proposals import GreedyTokenProposalConfig


def make_model(settings=None, sample_from_anchor=True):
    cfg = DSparkSpeculatorConfig(
        transformer_layer_config=Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=128,
            sliding_window=16,
            layer_types=["sliding_attention"] * 3,
            _attn_implementation="eager",
        ),
        draft_vocab_size=32,
        block_size=3,
        aux_hidden_state_layer_ids=[0, 1, 2],
        mask_token_id=31,
        markov_rank=4,
        intermediate_supervision=settings,
        sample_from_anchor=sample_from_anchor,
        speculators_config=SpeculatorsConfig(
            algorithm="dspark",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=None, architectures=["Qwen3ForCausalLM"]
            ),
        ),
    )
    model = DSparkDraftModel(cfg)
    for parameter in model.parameters():
        if torch.isnan(parameter).any():
            torch.nn.init.normal_(parameter, std=0.1)
    return model


def batch():
    return {
        "hidden_states": torch.randn(1, 12, 48),
        "input_ids": torch.randint(0, 31, (1, 12)),
        "loss_mask": torch.ones(1, 12),
        "verifier_last_hidden_states": torch.randn(1, 12, 16),
        "document_ids": torch.zeros(1, 12, dtype=torch.long),
        "max_anchors": 2,
    }


@pytest.mark.parametrize("sample", [True, False])
def test_real_forward_backward_and_schedule(sample):
    model = make_model({"rank": 4, "decay_steps": 2, "logit_chunk_size": 2}, sample)
    model.train()
    model.set_train_step(2)
    _, loss, metrics = model(**batch())
    assert torch.isfinite(loss)
    assert metrics["inter_aux_weight_sum"].item() == pytest.approx(0.08)
    loss.backward()
    for head in model.intermediate_heads:
        assert torch.isfinite(head.down.grad).all()
        assert head.down.grad.norm() > 0
    assert model.layers[0].mlp.up_proj.weight.grad.norm() > 0


def test_eval_and_stripped_export_have_identical_outputs(tmp_path):
    model = make_model({"rank": 4}).eval()
    inputs = batch()
    with patch.object(
        IntermediateHead, "forward", side_effect=AssertionError("eval head")
    ):
        torch.manual_seed(7)
        before = model._backbone_forward(**inputs)[1]
        model(**inputs)
    train_state = copy.deepcopy(model.state_dict())
    model.save_pretrained(tmp_path / "train")
    saved = DSparkSpeculatorConfig.from_pretrained(tmp_path / "train")
    assert saved.intermediate_supervision.rank == 4
    model.remove_intermediate_supervision()
    assert not any(k.startswith("intermediate_heads.") for k in model.state_dict())
    torch.manual_seed(7)
    after = model._backbone_forward(**inputs)[1]
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, train_state[key], rtol=0, atol=0)
    model.save_pretrained(tmp_path / "serve")
    assert (
        DSparkSpeculatorConfig.from_pretrained(
            tmp_path / "serve"
        ).intermediate_supervision
        is None
    )


def test_disabled_has_no_extra_parameters_or_loss():
    model = make_model()
    assert model.intermediate_heads is None
    _, loss, metrics = model(**batch())
    assert torch.isfinite(loss)
    assert not any(k.startswith("inter") for k in metrics)


def test_padding_and_teacher_alignment_match_direct_ce():
    settings = IntermediateSupervisionConfig(rank=4, logit_chunk_size=2)
    head = IntermediateHead(16, 4, 32, 1e-6)
    hidden = torch.randn(1, 6, 16, requires_grad=True)
    targets = torch.randn(1, 6, 32)
    mask = torch.tensor([[0, 1, 1, 0, 0, 0]], dtype=torch.float32)
    loss, _ = auxiliary_loss([head], [hidden], targets, mask, 3, 4, settings, False)
    code = head(hidden)
    ce = torch.nn.functional.cross_entropy(
        code @ head.codebook.T, targets.argmax(-1).flatten(), reduction="none"
    )
    expected = (ce[1] + ce[2] * torch.exp(torch.tensor(-0.25))) / 2
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert hidden.grad[0, [0, 3, 4, 5]].abs().sum() == 0
    zero, _ = auxiliary_loss([head], [hidden], targets, mask * 0, 3, 4, settings, False)
    assert zero.item() == 0


@pytest.mark.parametrize(
    "settings",
    [
        {"rank": 0},
        {"weight": float("nan")},
        {"decay_steps": 0},
        {"logit_chunk_size": 0},
    ],
)
def test_invalid_config_fails(settings):
    with pytest.raises(ValueError):
        IntermediateSupervisionConfig(**settings)
