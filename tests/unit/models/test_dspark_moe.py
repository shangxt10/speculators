"""Sparse FFN equivalence, routing gradients and DSpark checkpoint persistence."""

import pytest
import torch
from transformers import Qwen3Config

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.dspark.config import DSparkSpeculatorConfig
from speculators.models.dspark.core import DSparkDraftModel
from speculators.models.dspark.moe import DraftMoEConfig, DSparkMoE
from speculators.proposals import GreedyTokenProposalConfig


@pytest.mark.parametrize("score", ["softmax", "sigmoid", "sqrtsoftplus"])
@pytest.mark.parametrize("shared", [0, 1])
def test_reference_routing_and_gradients(score, shared):
    cfg = DraftMoEConfig(
        num_experts=4,
        experts_per_token=2,
        expert_intermediate_size=6,
        num_shared_experts=shared,
        score_function=score,
        routed_scaling_factor=1.5,
    )
    layer = DSparkMoE(8, cfg).double()
    x = torch.randn(2, 3, 8, dtype=torch.double, requires_grad=True)
    output = layer(x)
    flat = x.flatten(0, 1)
    logits = torch.nn.functional.linear(flat.float(), layer.gate.weight.float())
    scores = (
        logits.softmax(-1)
        if score == "softmax"
        else logits.sigmoid()
        if score == "sigmoid"
        else torch.nn.functional.softplus(logits).sqrt()
    )
    weights, ids = scores.topk(2, dim=-1)
    weights = weights / weights.sum(-1, keepdim=True) * 1.5
    expected = []
    for token, token_ids, token_weights in zip(flat, ids, weights, strict=True):
        total = torch.zeros_like(token)
        for expert, weight in zip(token_ids.tolist(), token_weights, strict=True):
            gate, up = (layer.experts.gate_up_proj[expert] @ token).chunk(2)
            total = total + weight * (
                layer.experts.down_proj[expert] @ (gate.sigmoid() * gate * up)
            )
        if shared:
            gate, up = (layer.shared_experts.gate_up_proj[0] @ token).chunk(2)
            total = total + layer.shared_experts.down_proj[0] @ (
                gate.sigmoid() * gate * up
            )
        expected.append(total)
    torch.testing.assert_close(output.flatten(0, 1), torch.stack(expected))
    output.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert layer.gate.weight.grad.norm() > 0
    assert layer.active_parameter_count() == 8 * 4 + (2 + shared) * 3 * 8 * 6


def make_model(moe=None):
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
        draft_moe=moe,
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


def test_full_model_backward_and_config_roundtrip(tmp_path):
    model = make_model(
        {"num_experts": 4, "experts_per_token": 2, "expert_intermediate_size": 8}
    )
    inputs = {
        "hidden_states": torch.randn(1, 12, 48),
        "input_ids": torch.randint(0, 31, (1, 12)),
        "loss_mask": torch.ones(1, 12),
        "verifier_last_hidden_states": torch.randn(1, 12, 16),
        "document_ids": torch.zeros(1, 12, dtype=torch.long),
        "max_anchors": 2,
    }
    _, loss, _ = model(**inputs)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.layers[0].mlp.experts.gate_up_proj.grad.norm() > 0
    assert model.layers[0].self_attn.q_proj.weight.grad.norm() > 0
    model.save_pretrained(tmp_path)
    cfg = DSparkSpeculatorConfig.from_pretrained(tmp_path)
    cfg.transformer_layer_config._attn_implementation = "eager"
    restored = DSparkDraftModel(cfg)
    restored.load_state_dict(model.state_dict())
    model.eval()
    restored.eval()
    torch.manual_seed(17)
    before = model._backbone_forward(**inputs)[1]
    torch.manual_seed(17)
    after = restored._backbone_forward(**inputs)[1]
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_default_dense_unchanged():
    model = make_model()
    assert model.config.draft_moe is None
    assert type(model.layers[0].mlp).__name__ == "Qwen3MLP"


def test_invalid_topk_rejected():
    with pytest.raises(ValueError, match="experts_per_token"):
        DraftMoEConfig(num_experts=2, experts_per_token=3)


def test_unselected_experts_and_empty_input():
    layer = DSparkMoE(
        8,
        DraftMoEConfig(
            num_experts=8,
            experts_per_token=2,
            expert_intermediate_size=4,
            num_shared_experts=0,
        ),
    )
    value = layer(torch.randn(1, 8, requires_grad=True))
    value.sum().backward()
    grad = layer.experts.gate_up_proj.grad.flatten(1).abs().sum(1)
    assert (grad > 0).sum() == 2
    empty = torch.empty(0, 8, requires_grad=True)
    layer(empty).sum().backward()
    assert empty.grad.shape == empty.shape
