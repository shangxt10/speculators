# Qwen3 draft 可选 MLA

Qwen3 draft 可以在 `draft_config.json` 的 `layer_types` 中用 `"mla"` 指定
Multi-head Latent Attention。支持 `dspark`、`dflash`、`eagle3` 和 `peagle`。
这里只改变 draft，target/verifier 无需改动。

未指定 `"mla"` 且未开启旧版全局 MLA 开关时，保留原 attention 和参数名称。
旧 checkpoint 也沿用原结构。新 checkpoint 保存层类型及 MLA 维度，恢复训练或离线
评测时自动重建相同结构。

## 启用方式

将五层 draft 配置中的 `layer_types` 替换为：

```json
"layer_types": ["mla", "mla", "mla", "mla", "mla"]
```

训练命令仍使用 `--draft-config /path/to/draft_config.json`，无需再加
`--draft-use-mla`，也无需写 `"use_mla": true`。其余模型配置可以保持不变。
完整的 3584 hidden size、5 层示例见
[qwen3_3584_mla.json](../../../examples/train/configs/qwen3_3584_mla.json)。

按层选择也受支持，例如 `"layer_types": ["full_attention", "mla", "mla",
"full_attention", "mla"]`。`"mla"` 使用 full-attention 的 mask 规则，普通层仍使用
原 attention；混合层各自采用正确维度的 RoPE。列表长度必须等于 `num_hidden_layers`。
当列表包含 `"mla"` 时，以逐层列表为准，旧版 `use_mla` 开关不会把其余层也切为 MLA。

下面的命令行方式保留兼容性，适用于**未使用 `--draft-config`** 的训练：

在已有的 `scripts/train.py` 或 `torchrun ... scripts/train.py` 命令中增加：

```bash
--draft-use-mla
```

DSpark/DFlash/P-EAGLE 默认 draft 架构为 Qwen3；EAGLE3 需要同时指定
`--draft-arch qwen3`。如需自定义维度，可以使用：

```bash
--draft-use-mla \
--mla-kv-lora-rank 256 \
--mla-q-lora-rank 512 \
--mla-qk-nope-head-dim 64 \
--mla-qk-rope-head-dim 64 \
--mla-v-head-dim 128
```

| 参数 | 含义 | 不指定时 |
| --- | --- | --- |
| `--draft-use-mla` | 启用 MLA | 关闭 |
| `--mla-kv-lora-rank` | 所有 head 共享的 KV latent 宽度 | `min(512, max(1, hidden_size // 4))` |
| `--mla-q-lora-rank` | Query 的可选低秩中间维度 | 直接投影 Query，不压缩 |
| `--mla-qk-nope-head-dim` | 每个 head 不使用 RoPE 的 Q/K 宽度 | `max(1, head_dim - 默认 RoPE 宽度)` |
| `--mla-qk-rope-head-dim` | 每个 Query head 的 RoPE 宽度，Key 共享该分支 | `max(2, (head_dim // 4) * 2)` |
| `--mla-v-head-dim` | 每个 head 的 Value 宽度 | 原 `head_dim` |

维度必须为正整数，RoPE 宽度必须为偶数。单独传维度但不启用 MLA 会报错。
`--draft-attn-impl` 继续选择 `simple_flex_attention`、`sdpa` 或 `eager`。
Ascend 上可以沿用 `--draft-attn-impl sdpa`；实际 NPU 训练和性能需要在设备上验证。

## 五层全部为 SWA + MLA

`"mla"` 标签固定使用 full-attention 的 mask 规则，不能与滑动窗口叠加。要让五层
draft 全部是滑动窗口注意力且使用 MLA，应把 `layer_types` 全部写成
`"sliding_attention"`，并用 `use_mla` 开启 MLA（`layer_types` 中没有 `"mla"`
时，MLA 判定回退到这个全局开关）：

```json
{
  "model_type": "qwen3",
  "vocab_size": 131072,
  "hidden_size": 3584,
  "intermediate_size": 18432,
  "num_hidden_layers": 5,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "hidden_act": "silu",
  "max_position_embeddings": 262144,
  "rms_norm_eps": 1e-06,
  "rope_theta": 10000,
  "use_sliding_window": true,
  "sliding_window": 2048,
  "layer_types": [
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention"
  ],
  "use_mla": true,
  "mla_kv_lora_rank": 512,
  "mla_q_lora_rank": null,
  "mla_qk_nope_head_dim": 64,
  "mla_qk_rope_head_dim": 64,
  "mla_v_head_dim": 128
}
```

把上面的内容保存为文件后，用 `--draft-config /path/to/your_config.json` 训练即可，
不需要再传 `--draft-use-mla` 或 `--mla-*`；这些维度字段会随模型写入 checkpoint，
恢复训练和离线评测时按 `layer_types` 重建相同的 MLA + SWA 结构。

与实现对应的几个要点：

- 滑动窗口取值：每个 `Qwen3MLAAttention` 在自己的层标签为 `"sliding_attention"`
  时直接使用 `config.sliding_window` 作为 mask 窗口，DSpark/DFlash 也按
  `layer_types` 选出 `sliding_window_indices` 并给这些层传滑动窗口 mask。
  draft 不读取 `use_sliding_window`，它只为与 HF 配置保持一致，真正生效的是
  正整数 `sliding_window`；写成 `null` 会让这些层退化为 full attention。
- MLA 维度：示例中的数值与默认推导一致（KV latent 512、NoPE 64、RoPE 64、
  Value 128）；`"mla_q_lora_rank": null` 表示 Query 直接投影，不做低秩压缩。
  省略这些字段时 `configure_mla` 会按同样的规则补齐并保存。
- RoPE：五层全部使用 MLA，因此共享一份按 `mla_qk_rope_head_dim` 生成的
  MLA RoPE，不再逐层创建 `Qwen3RotaryEmbedding`。
- 命令行等价写法（不使用 `--draft-config`）为
  `--draft-use-mla --sliding-window 2048`，并保持
  `--full-attention-indices` 为默认空列表（默认即把所有 draft 层标为
  `sliding_attention`）。

## 配置文件和 checkpoint

遵循仓库已有初始化规则，`--draft-config`、`--from-pretrained` 不能与
decoder 结构参数（包括新的 MLA 开关、维度）同时使用。
使用 `--draft-config` 时，可以在层类型之外添加可选维度字段；
完整 speculator 配置中则放在 `transformer_layer_config` 下：

```json
{
  "model_type": "qwen3",
  "num_hidden_layers": 5,
  "layer_types": ["mla", "mla", "mla", "mla", "mla"],
  "mla_kv_lora_rank": 256,
  "mla_q_lora_rank": 512,
  "mla_qk_nope_head_dim": 64,
  "mla_qk_rope_head_dim": 64,
  "mla_v_head_dim": 128
}
```

这是新增字段示例，应保留原配置中的 `hidden_size`、层数、词表等字段。
不填写维度字段时，使用上表默认值并随模型保存。对于上述 3584/128 配置，默认
KV rank=512、NoPE=64、RoPE=64、Value=128，Query 使用直接投影。
旧的 `use_mla=true` 写法仍然支持：当列表中没有显式 `"mla"` 时，它将所有层切换为 MLA。
Python 的 `from_training_args` 入口对应参数为 `draft_use_mla=True`、
`mla_kv_lora_rank=...` 等。

MLA 使用不同的投影权重，不能通过修改旧 GQA checkpoint 的配置无损转换权重。
应初始化并训练新 MLA draft，或加载已经训练的 MLA checkpoint。

## 结构和执行范围

实现采用 [DeepSeek-V2 的 MLA 结构](https://arxiv.org/abs/2405.04434)：
KV 联合低秩压缩、latent RMSNorm、按 head 上投影，以及独立的共享 RoPE Key。
只有位置分支参与 RoPE；attention scale 使用内容宽度与 RoPE 宽度之和。
原 Qwen3 的逐 head Q/K RMSNorm 被 latent RMSNorm 替代。

DSpark/DFlash 仍然将 target hidden states 和 draft tokens 拼为 KV 序列，
保留已有的 document、anchor/block、sliding-window 和 full-attention mask。
EAGLE3/P-EAGLE 首层仍接受两倍 hidden size 的拼接输入。

传入 `DynamicCache` 时，每层每个 token 仅缓存 `mla_kv_lora_rank +
mla_qk_rope_head_dim` 个元素（latent 和已旋转的共享 Key），支持追加和裁剪。
计算 attention 时仍展开成各 head 的 K/V，复用现有后端。Flex 路径会将 head
宽度补零到适合 kernel 的二次幂，并保持未补零时的 scale。

本适配覆盖仓库内训练和使用相同 draft 类的离线评测。尚未实现矩阵吸收或专用
MLA decode kernel，也未适配外部 vLLM/SGLang 的 draft loader；不能据此承诺
端到端加速或直接部署到这些引擎。缓存压缩比例也取决于所选 latent 维度和原 GQA 配置。

## DSpark 离线推理测评（含树投机）

两套入口都从 **训练得到的 checkpoint 目录** 读取 `config.json` 中的
`transformer_layer_config.layer_types` 和 MLA 维度，并实例化对应的 attention：

- `scripts/evaluate/dspark_offline_eval.py`：普通离线评测，支持线性逐 token
  验证及树投机。
- `xingchen4_eval_ep/evaluate_ep.py`：Ascend expert parallel 离线评测，复用相同
  draft 加载和解码逻辑；NPU 拼接兼容处理同时支持 GQA、全 MLA 和混合层。

评测时不需要再传 `--draft-config` 或 MLA 开关。`--draft-model` 必须指向完整
MLA checkpoint，不能指向训练用的 `qwen3_3584_mla.json`，也不能只修改旧 GQA
checkpoint 的层类型。`--draft-attn-impl sdpa` 选择计算后端，与 MLA 结构可以同时使用。
`--dtype bfloat16` 同时用于加载 target 和 draft；加载 verifier 共享权重时使用本次
`--verifier-model` 指定的路径，避免继续引用 checkpoint 中旧机器上的训练路径。

可复制运行示例 [dspark_mla_offline.sh](../../../examples/evaluate/dspark_mla_offline.sh)：

```bash
# 在 speculators 仓库根目录执行，替换下面三个路径。
export MODEL=/path/to/target
export DRAFT_MODEL=/path/to/trained_mla_checkpoint
export DATASETS_ROOT=/path/to/eval_jsonl

# 普通离线树投机：target 必须能在所选单设备上加载。
DEVICE=npu:0 EVAL_MODE=tree \
  bash examples/evaluate/dspark_mla_offline.sh

# 单机 16 卡 EP 树投机：使用已有 EP evaluator 支持的 MoE target。
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  EVAL_BACKEND=ep EP_SIZE=16 EVAL_MODE=tree \
  bash examples/evaluate/dspark_mla_offline.sh
```

示例默认开启 `--measure-base-speedup`，测量同一 target 的自回归基线。
`EVAL_MODE=token` 使用原线性逐 token 验证；
`EVAL_MODE=tree` 默认添加 `--tree-top-k 4 --tree-max-nodes 32 --temperature 0`。
可用 `TREE_TOP_K`、`TREE_MAX_NODES`、`DATASETS`、`OUTPUT_DIR` 环境变量调整。
现有 CorrectionHead 和 candidate selector 的树模式限制仍然适用；
目前支持 plain/Markov DSpark。

使用自己的启动脚本时，保留原入口和设备分配，指定 MLA checkpoint，并增加：

```bash
--draft-model /path/to/trained_mla_checkpoint \
--draft-attn-impl sdpa \
--dtype bfloat16 \
--temperature 0 \
--tree-top-k 4 \
--tree-max-nodes 32
```

启动日志逐层输出实际类名，默认 3584/128 MLA 配置应显示类似：

```text
Draft layer 0: Qwen3MLAAttention backend=sdpa dtype=torch.bfloat16 kv_rank=512 nope_dim=64 rope_dim=64 v_dim=128
```

混合层会同时显示 `Qwen3MLAAttention` 和 `Qwen3DFlashAttention`。这确认实际实例化
的结构；运行成功及速度仍需要查看后续生成结果和评测指标。

树分支复用一次 MLA draft backbone 前向的结果。target 的树 mask、位置编号、
接受路径及 KV cache 裁剪规则保持原样；这里的 target cache 与 MLA draft latent
cache 是不同对象。当前离线 evaluator 每轮重算 draft 的上下文投影，没有启用跨轮
MLA latent cache 或矩阵吸收，因此不能直接将训练结构适配解释为已有 MLA 推理加速。

CPU 回归覆盖全 MLA/混合层 checkpoint 加载、float32/bfloat16 权重类型、两种
`sample_from_anchor` 设置、两种验证模式的贪心输出，以及 EP 拼接补丁前后 draft
hidden states/logits 一致性。Ascend 实机、EP collective 和吞吐性能需在部署环境验证。
