#!/usr/bin/env python3
"""Run one prompt through a trained Speculators DSpark checkpoint."""

import argparse
import json
from pathlib import Path

import openai
import torch
from transformers import AutoProcessor

from hs_connectors import FileTransfer
from speculators.data_generation.preprocessing import get_tokenizer
from speculators.data_generation.vllm_client import extract_output
from speculators.model import SpeculatorModel


DEFAULT_PROMPT = """Answer the following math problem step by step and put the \
final answer after ####.

Natalia sold clips to 48 of her friends in April, and then she sold half as many \
clips in May. How many clips did Natalia sell altogether in April and May?"""


class TraceWriter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

    def tensor(self, name: str, value: torch.Tensor) -> None:
        cpu = value.detach().to("cpu").contiguous()
        torch.save(cpu, self.output_dir / f"{name}.pt")
        flat = cpu.reshape(-1)
        summary: dict[str, object] = {
            "shape": list(cpu.shape),
            "dtype": str(cpu.dtype),
            "numel": cpu.numel(),
            "head": flat[:8].tolist(),
            "tail": flat[-8:].tolist(),
        }
        if cpu.is_floating_point() and cpu.numel() > 0:
            finite = cpu.float()[torch.isfinite(cpu)]
            summary["finite"] = int(finite.numel())
            if finite.numel() > 0:
                summary.update(
                    min=float(finite.min()),
                    max=float(finite.max()),
                    mean=float(finite.mean()),
                    std=float(finite.std(unbiased=False)),
                    l2=float(torch.linalg.vector_norm(finite)),
                )
            if cpu.ndim > 0 and cpu.shape[-1] >= 8:
                rows = cpu.float().reshape(-1, cpu.shape[-1])
                top_values, top_indices = rows.topk(5, dim=-1)
                summary["first_row_top5_ids"] = top_indices[0].tolist()
                summary["first_row_top5_values"] = top_values[0].tolist()
                summary["last_row_top5_ids"] = top_indices[-1].tolist()
                summary["last_row_top5_values"] = top_values[-1].tolist()
        (self.output_dir / f"{name}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[Speculators DSpark parity] {name}={summary}")


def _tokenize_prompt(processor, prompt: str) -> torch.Tensor:
    encoded = processor.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    if encoded.ndim == 2:
        encoded = encoded[0]
    return encoded.long()


def _load_hidden_states(
    endpoint: str,
    hidden_states_path: Path,
    input_ids: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], int]:
    client = openai.OpenAI(base_url=endpoint, api_key="EMPTY", max_retries=0)
    model_id = client.models.list().data[0].id
    response = client.completions.create(
        model=model_id,
        prompt=input_ids.tolist(),
        max_tokens=1,
        temperature=0,
        extra_body={"return_token_ids": True},
    )
    handle = extract_output(response, input_ids.tolist())
    completion_ids = getattr(response.choices[0], "token_ids", None)
    if completion_ids is None and response.choices[0].model_extra is not None:
        completion_ids = response.choices[0].model_extra.get("token_ids")
    if not completion_ids:
        raise RuntimeError(
            "Hidden-state service response did not include completion token_ids"
        )
    payload = FileTransfer(hidden_states_path).get_generated(handle)
    if payload is None:
        raise RuntimeError(f"Hidden-state service returned unreadable handle: {handle}")
    if not torch.equal(payload["token_ids"].long(), input_ids):
        raise RuntimeError("Hidden-state service returned different prompt token ids")
    return payload, int(completion_ids[0])


def _move_model(model, device: str) -> None:
    if device.startswith("npu"):
        import torch_npu  # noqa: F401, PLC0415

    model.to(device)
    model.eval()


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir
    writer = TraceWriter(output_dir)
    processor = AutoProcessor.from_pretrained(
        args.verifier,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = get_tokenizer(processor)
    input_ids_cpu = _tokenize_prompt(processor, args.prompt)
    payload, target_next_token = _load_hidden_states(
        args.extract_endpoint,
        args.hidden_states_path,
        input_ids_cpu,
    )

    model = SpeculatorModel.from_pretrained(
        args.draft_model,
        local_files_only=True,
    )
    if model.__class__.__name__ != "DSparkDraftModel":
        raise TypeError(f"Expected DSparkDraftModel, got {model.__class__.__name__}")
    if model.config.markov_head_type != "vanilla":
        raise NotImplementedError(
            "vLLM parity rollout currently supports vanilla Markov only"
        )
    _move_model(model, args.device)

    hidden_all = payload["hidden_states"]
    layer_ids = list(model.target_layer_ids)
    expected_layers = len(layer_ids) + 1
    if hidden_all.ndim != 3 or hidden_all.shape[1] != expected_layers:
        raise ValueError(
            f"Expected hidden_states [T,{expected_layers},H], "
            f"got {list(hidden_all.shape)}"
        )

    for index, layer_id in enumerate(layer_ids):
        print(
            f"[Speculators DSpark parity] aux_hidden_{index} "
            f"uses checkpoint layer {layer_id}"
        )
        writer.tensor(f"aux_hidden_{index}", hidden_all[:, index])
    aux_concat_cpu = hidden_all[:, :-1].flatten(1)
    writer.tensor("aux_hidden_concat", aux_concat_cpu)
    writer.tensor("target_token_ids", input_ids_cpu)
    writer.tensor("target_positions", torch.arange(input_ids_cpu.numel()))
    writer.tensor("next_token_ids", torch.tensor([target_next_token]))

    device = torch.device(args.device)
    dtype = next(model.parameters()).dtype
    prompt_length = input_ids_cpu.numel()
    block_size = int(model.block_size)
    hidden_size = hidden_all.shape[-1]
    aux_width = aux_concat_cpu.shape[-1]

    aux_concat = torch.zeros(
        1,
        prompt_length + 1 + block_size,
        aux_width,
        dtype=dtype,
        device=device,
    )
    aux_concat[0, :prompt_length].copy_(aux_concat_cpu.to(device=device, dtype=dtype))
    last_hidden = torch.zeros(
        1,
        prompt_length + 1 + block_size,
        hidden_size,
        dtype=dtype,
        device=device,
    )
    last_hidden[0, :prompt_length].copy_(
        hidden_all[:, -1].to(device=device, dtype=dtype)
    )
    padded_ids = torch.zeros(
        1, prompt_length + 1 + block_size, dtype=torch.long, device=device
    )
    padded_ids[0, :prompt_length].copy_(input_ids_cpu.to(device))
    padded_ids[0, prompt_length] = target_next_token
    loss_mask = torch.zeros_like(padded_ids, dtype=torch.bool)
    loss_mask[0, prompt_length] = True
    position_ids = torch.arange(
        prompt_length + 1 + block_size, dtype=torch.long, device=device
    ).unsqueeze(0)
    document_ids = torch.zeros_like(padded_ids)

    fc_output = model.fc(aux_concat)
    fc_hidden_norm = model.hidden_norm(fc_output)
    writer.tensor("fc_output", fc_output[:, :prompt_length])
    writer.tensor("fc_hidden_norm", fc_hidden_norm[:, :prompt_length])

    hidden, base_logits_flat, _, _, anchors = model._backbone_forward(
        aux_concat,
        padded_ids,
        loss_mask,
        last_hidden,
        document_ids,
        position_ids,
        max_anchors=1,
    )
    expected_anchor = prompt_length
    if int(anchors[0]) != expected_anchor:
        raise RuntimeError(f"Expected anchor {expected_anchor}, got {int(anchors[0])}")
    base_logits = base_logits_flat.view(1, block_size, -1)
    writer.tensor("base_logits", base_logits)

    previous_target_id = padded_ids[:, expected_anchor]
    mapped_ids = []
    for index in range(block_size):
        writer.tensor(f"markov_input_{index}", previous_target_id)
        markov_embedding = model.markov_head.prev_embeddings(previous_target_id)
        writer.tensor(f"markov_embedding_{index}", markov_embedding)
        markov_bias = model.markov_head.block_bias(
            prev_token_ids=previous_target_id[:, None],
            hidden_states=hidden[:, index : index + 1],
            prev_emb=markov_embedding[:, None],
        )[:, 0]
        writer.tensor(f"markov_bias_{index}", markov_bias)
        writer.tensor(f"base_logits_{index}", base_logits[:, index])
        biased_logits = base_logits[:, index] + markov_bias
        writer.tensor(f"biased_logits_{index}", biased_logits)
        draft_argmax = biased_logits.argmax(dim=-1)
        writer.tensor(f"draft_argmax_{index}", draft_argmax)
        if model.d2t is None:
            mapped_argmax = draft_argmax
        else:
            mapped_argmax = draft_argmax + model.d2t[draft_argmax]
        writer.tensor(f"mapped_argmax_{index}", mapped_argmax)
        mapped_ids.append(mapped_argmax)
        previous_target_id = mapped_argmax

    mapped = torch.stack(mapped_ids, dim=1)
    writer.tensor("mapped_draft_token_ids", mapped)
    decoded = tokenizer.decode(mapped[0].tolist(), skip_special_tokens=False)
    print(f"Prompt token count: {prompt_length}")
    print(
        f"Target next token: {target_next_token} "
        f"({tokenizer.decode([target_next_token])!r})"
    )
    print(f"Mapped draft ids: {mapped[0].tolist()}")
    print(f"Mapped draft text: {decoded!r}")
    print(f"Trace saved to: {output_dir}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--verifier", required=True)
    parser.add_argument("--extract-endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--hidden-states-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
