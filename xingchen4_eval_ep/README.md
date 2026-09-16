# Pure EP evaluation for Xingchen4 + DSpark

This directory is a standalone pure-PyTorch/HCCL expert-parallel evaluator. It
does not use vLLM and does not modify the existing TP or offline evaluator.

Only the verifier's routed experts are sharded. Attention, embeddings, norms,
dense MLPs, shared experts, the output head, and the DSpark draft model remain
replicated on every rank. The evaluator refuses an EP plan containing ordinary
`colwise` or `rowwise` TP rules.

Two verifier expert layouts are supported:

- fused `experts.gate_up_proj/down_proj`: Transformers native grouped-GEMM EP;
- per-expert `experts.<id>.gate_proj/up_proj/down_proj`: a custom ModuleList EP
  backend keeps only the rank-local expert range and performs one fixed-shape
  HCCL all-reduce after each routed MoE block.

## Requirements

- Launch with `torchrun`, one process per visible NPU.
- Transformers must provide `DistributedConfig`, `ep_router`, `grouped_gemm`,
  and `moe_tp_experts`.
- `accelerate>=1.1.0` must be installed because Transformers places replicated
  and sharded weights on the rank-local NPU while loading.
- The verifier config must define `base_model_ep_plan`.
- The EP size must evenly divide the routed-expert count.
- Greedy decoding (`--temperature 0`) is required so every rank enters the same
  model collectives in the same order.

On Ascend, target-context and draft-noise projections may carry different
physical formats (NCHW versus NCL). The EP evaluator replaces only the in-memory
`Qwen3DFlashAttention.forward` methods and builds K/V through copies into an ND
buffer instead of `torch.cat`. The installed Speculators package is not edited.

Xingchen4 also keeps four mHC residual streams inside each verifier layer, so
Transformers records intermediate states as `[batch, sequence, hc_mult, hidden]`.
Before passing selected layers to DSpark, this evaluator uses the verifier's own
learned `hc_head`/`hyper_head` to collapse them to `[batch, sequence, hidden]`.
It then validates the concatenated width against `draft.fc.in_features`. Do not
replace this learned collapse with a mean or a fixed stream selection: that
would no longer match the hidden-state distribution used for draft training.
Private Xingchen variants sometimes rename this module. If no conventional name
is present, the first verifier forward temporarily probes module input/output
shapes and remembers the learned module that performs the direct
`[B, S, hc_mult, H] -> [B, S, H]` transition; the temporary hooks are removed
immediately after discovery. Both tensor-returning heads and tuple-returning mHC
interfaces such as `(post, comb, collapsed)` are supported; tuple outputs are
accepted only when exactly one element matches `[B, S, H]`.

Check the local model config without loading weights:

```bash
python3 scripts/evaluate/ep_eval/inspect_ep_support.py \
  --model /hpfs/model/xingchen4 \
  --expert-parallel-size 4 \
  --trust-remote-code \
  --local-files-only
```

## EP4 smoke test

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

torchrun \
  --standalone \
  --nproc-per-node=4 \
  scripts/evaluate/ep_eval/evaluate_ep.py \
  --verifier-model /hpfs/nlp/yangzhihao/ckpts/TeleChat4-29B/29b_12.05T_wodsa_id20_s3_256k_8_260903_1000_final_hf \
  --draft-model /hpfs/huawei-2607/shangxt/Telecom-29B/telecom_80w_no_thinking_0910_16_16/checkpoints/checkpoint_best \
  --datasets-root /hpfs/huawei-2607/shangxt/evaldata \
  --datasets aime25 \
  --output-dir ./eval_results/xingchen4_pytorch_ep4_smoke \
  --expert-parallel-size 4 \
  --max-samples 1 \
  --max-new-tokens 32 \
  --temperature 0 \
  --dtype bfloat16 \
  --draft-attn-impl sdpa \
  --enable-thinking false \
  --raw-prompt-mode auto \
  --trust-remote-code \
  --local-files-only
```

Do not pass `--tensor-parallel-size` or `--ascend-devices`. `torchrun` owns
process creation and `ASCEND_RT_VISIBLE_DEVICES` selects the physical NPUs.

For the supplied Xingchen4 config (`num_experts=64`), EP4 loads 16 routed
experts per rank. After loading, the evaluator also verifies that Transformers
selected the EP plan and did not activate ordinary tensor-parallel rules.
Router/expert-container rules are verified through module hooks, while
`grouped_gemm` is verified from the locally loaded parameter shape because it
is a parameter-sharding rule rather than a module hook.
