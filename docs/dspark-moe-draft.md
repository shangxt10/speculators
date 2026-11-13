# Optional MoE FFN for DSpark

This change replaces only the DSpark draft's dense FFN with routed/shared SwiGLU experts. Attention, RoPE, target-HS conditioning, Markov/confidence heads, loss alignment and the offline evaluation interface remain the existing DSpark code. It does not import the DSV4 attention or mHC architecture.

## Try with your existing training command

From a clean checkout of the customer repository, fetch this PR:

```bash
git fetch https://github.com/shangxt10/speculators.git pull/2/head:review/dspark-moe
git switch review/dspark-moe
```

Use your existing DSpark training environment and native `scripts/train.py` command. If the package was installed editable from this checkout, switching branches updates the imported code; otherwise reinstall from this checkout using your usual installation procedure. Keep your model/data/HS paths, attention settings and distributed launcher.

The following is an **argument fragment**, not a standalone shell command. It maps the expert dimensions and routing hyperparameters of our three-layer DSV4 draft to this PR's existing customer-attention model. For a fresh experiment, add or replace:

```bash
--speculator-type dspark --num-layers 3 \
--draft-moe '{"num_experts":256,"experts_per_token":6,"expert_intermediate_size":2048,"num_shared_experts":1,"score_function":"sqrtsoftplus","normalize_topk":true,"routed_scaling_factor":1.5}'
```

This is a **reference experiment configuration**, not the configuration of the 100-step native validation below. That functional validation used 3 layers, 64 routed experts, top-4, expert width 1024, one shared expert, softmax scores and scaling 1.0. The 256/top-6/2048 configuration still requires full-size validation in this customer implementation; the smaller run does not establish its memory use, training stability or speed.

Use a new, unused `--save-path` and a separate `--log-dir` for this experiment: the trainer automatically resumes checkpoints in its save directory. Replace an existing `--num-layers 5` with `--num-layers 3`, rather than leaving both. If your decoder comes from `--draft-config`, set `num_hidden_layers` to 3 in that decoder config and omit `--num-layers`; those options are mutually exclusive.

This fragment uses fresh model initialization. If your command uses `--from-pretrained`, do not append `--draft-moe` to it: the saved model config defines its FFN, and the CLI rejects a conflicting override. When loading an already initialized MoE checkpoint, use `--from-pretrained` alone for its architecture. Resume training with the same save directory and original training/scheduler recipe to retain optimizer and global-step state; `--from-pretrained` by itself only loads model initialization.

For a quick initialization check, add `--dry-run` with a separate unused `--save-path`. This saves the initialized model and exits; it does not test backward or optimizer restoration. The 50+50 native training validation below covers those paths. The first-run config should contain `draft_moe`, with `transformer_layer_config.num_hidden_layers` equal to 3.

Adjust the expert budget explicitly before comparing a three-layer MoE to a five-layer dense draft. For width H and expert width I, the active FFN parameters per layer are `H*E + 3*H*I*(K+S)`, including the router, K selected routed experts and S shared experts. Keeping H and all per-layer settings fixed while moving from three to five layers multiplies the active FFN budget by 5/3. A five-layer model can retain the same MoE settings, but a comparison at equal active-parameter budget would require changing the expert width and/or selected expert count. Reducing E alone mainly reduces stored expert parameters, not the selected-expert compute. Attention and prediction heads must also be included when comparing whole-model budgets. This is not a latency estimate.

Matching these dimensions and score settings does not reproduce the complete DSV4 draft. This PR retains the customer's attention and does not include DSV4 mHC, its expert warm start, SwiGLU magnitude clamp or adaptive router-selection bias. The existing portable MoE implementation is unchanged by this reference configuration.

The router supports softmax, sigmoid and sqrt(softplus), optional top-k normalization and output scaling. Expert matrices are packed into a small number of parameters. The portable reference dispatch evaluates selected token/expert pairs without token dropping or a dense token-by-expert activation tensor. It is intended for correctness and initial training validation, not a claim of fused-kernel performance. There is no draft expert-parallel implementation in this PR; the repository's existing layer FSDP can shard these parameters.

Initialization follows the existing DSpark training path: target embeddings and output head are loaded as before; the new router and experts are randomly initialized. This PR does not silently copy the target's MLA weights into the draft's MHA/GQA projections. A target-expert warm start would need an explicit, shape-checked checkpoint mapping and a separate validation result.

The MoE recipe and expert weights are saved in the normal draft checkpoint. `--from-pretrained` restores that architecture and cannot be combined with `--draft-moe`. A separate vLLM serving adapter is needed before using the new packed expert layout in an external vLLM implementation; this change targets the repository's native DSpark model and offline evaluator.

```bash
PYTHONPATH=src:hs_connectors/src python -m pytest tests/unit/models/test_dspark_moe.py
```

Tests compare routing against an independently written per-token reference, check input/router/expert gradients, active-parameter accounting, an actual DSpark forward/backward, save/reload parity, and the unchanged dense default.

## Native training validation

On 2026-11-11, commit `d88124e172340ae9158ad8c37c345bde29f451ab` passed a 16-Ascend-NPU test using an independently installed checkout and the native `scripts/train.py`: train 50 optimizer steps, save, then start a new process and resume to 100. Import and dependency audits passed. Both checkpoints had finite model and optimizer tensors; optimizer and scheduler counters matched 50 and 100. The new expert parameters had nonzero optimizer moments. The first resumed LR was `5.33e-5` in the rounded log, matching the saved `5.327015646150716e-5`.

Training loss was finite throughout (first/last logged values: 39.564 / 3.432). The test used a 3-layer MoE with **64 routed experts, top-4, expert width 1024, one shared expert, softmax scores and scaling 1.0**, at hidden width 4096, vocabulary 129280, sequence length 256 and block size 7, with 16 distinct training samples repeated to 800 rows and 16 held-out functional samples. These are locally available DSV4 hidden states, not the customer's private checkpoint/data. Native teacher normalization also differs from the original DSV4 benchmark. This establishes the training, saving and recovery path, not acceptance length, throughput or quality gains.

The [sanitized validation summary](dspark-moe-validation.json) records exact versions, counters and source-receipt hashes. This branch also includes a small checkpoint fix: restoring scheduler state must restore each optimizer group's LR before the first resumed update. A regression test checks exact equality with an uninterrupted update.

Compressed-HS and site-specific dataset adapters are outside this PR.
