# DSpark 离线树投机适配说明

## 目标和范围

本适配参考 `TreeSpark-main` 的顺序离线推理实现，把树形候选构造和单次
target 树验证接入 `scripts/evaluate/dspark_offline_eval.py`。它只修改离线推理，
不包含 TreeSpark 的树一致性训练目标。

来源与许可说明见 [`TREESPARK_NOTICE.md`](TREESPARK_NOTICE.md)。

默认行为保持不变：`--tree-top-k 1` 使用原有 DSpark 线性投机；设置
`--tree-top-k >= 2` 才启用树投机。普通 DSpark checkpoint 可以直接用于树推理，
但没有经过 TreeSpark 树目标训练时，候选排序和接受长度可能低于树训练模型。

## TreeSpark 到 Speculators 的映射

TreeSpark 的一次树投机轮次由四步组成：

1. **一次 draft backbone 前向。** 当前 Speculators 的 `_single_anchor_backbone`
   继续产生整个 block 的 hidden states 和每个深度的基础 logits。树的分支不会
   重复运行 backbone。
2. **父节点条件展开。** 深度 `d` 的所有 frontier 父节点共享基础预测
   `L_d`，分别计算 `L_d + Markov(parent)`。每个父节点保留 top-k 子节点，
   然后按累计路径 log-probability 把活动 frontier 裁回 top-k。
3. **全局节点预算。** 把所有深度产生的候选按路径分数排序，在
   `--tree-max-nodes` 预算下打包。路径分数沿深度单调不增，并采用浅层优先的
   稳定 tie-break，所以打包结果保持前缀闭合。
4. **一次 target 树验证。** anchor 和全部打包节点组成一个 target batch
   序列。4D bool attention mask 让每个节点只看到已验证上下文、自身和祖先；
   同深度兄弟使用相同 position id。验证后从 anchor 开始沿 target argmax
   匹配子节点，提交最长匹配路径和一个 target bonus token。

## DSpark 兼容处理

### Markov head

- `vanilla`：为每个父 token 单独计算低秩 logit bias。
- `gated`：父 token bias 还使用当前深度的 backbone hidden state。
- `rnn`：每个树节点携带自己的 recurrent state；同一父节点的子节点继承该父
  节点完成当前 step 后的 state。
- `markov_rank=0`：仍可构树，但各父节点共享相同 logits，分支收益通常较低。

当 `sample_from_anchor=false` 时，slot 0 不产生 proposal，但 RNN 训练仍处理该
slot。树构造会先用 anchor 重放保留 slot，再开始深度 0 的分叉。

### 裁剪词表

候选 top-k 和路径概率在 draft 词表中计算；树节点在写入 proposal 前通过
`d2t` 转换为 target token id。Markov `W1` 始终接收 target/verifier token id，
与当前 DSpark 训练定义一致。

### KV cache 和 hidden states

target forward 会暂时把整棵 packed tree 追加到 `DynamicCache`。接受遍历完成后，
cache 只保留历史前缀、anchor 和接受路径节点。用于下一轮 DSpark 的 target
hidden states 也从树 forward 中按相同路径直接 gather，不会重跑接受路径。

### EP 模式

`xingchen4_eval_ep/evaluate_ep.py` 复用相同 evaluator。所有 EP rank 都构造相同
树、传入相同 tree mask，并以相同顺序进入 verifier collective。EP 模式原本就
要求 `--temperature 0`，因此满足树验证条件。

## 命令行

普通离线评测：

```bash
python scripts/evaluate/dspark_offline_eval.py \
  --verifier-model /path/to/verifier \
  --draft-model /path/to/dspark \
  --datasets-root /path/to/evaldata \
  --datasets gsm8k,math500 \
  --output-dir eval_results/tree \
  --temperature 0 \
  --tree-top-k 4 \
  --tree-max-nodes 32
```

Xingchen4 expert-parallel 离线评测只需在现有命令中增加：

```bash
--tree-top-k 4 \
--tree-max-nodes 32
```

恢复线性 DSpark：

```bash
--tree-top-k 1
```

## 限制

- 树模式只支持 `--temperature 0`。采样 target 需要递归 rejection sampling，
  当前实现不会用 greedy tree 近似替代。
- 树模式支持 plain/Markov DSpark，不与 `CorrectionHead` 或
  `candidate_selector` 同时启用。
- confidence head 在树模式中不参与截断，节点预算由 `tree_top_k` 和
  `tree_max_nodes` 决定。
- target 模型必须接受带 past KV 的 4D bool attention mask。代码已按
  TreeSpark 的 Transformers/SDPA 接口实现；私有 remote-code 模型仍需在实际
  NPU 环境验证其 attention mask 语义。

## 指标解释

- `draft_length`：树模式下表示每轮实际送入 target 的非根节点数，通常接近
  `tree_max_nodes`；线性模式下仍表示链长度。
- `accepted_draft_length`：沿 target greedy 路径接受的树深度。
- `acceptance_length`：`1 + accepted_draft_length`，其中 1 是 target bonus token。
- `position_accept_rates`：按树深度统计，不把 packed 节点序号误当成生成位置。
- `position_accept_prob_means` 和 `position_support_accept_rate_means`：树模式没有
  rejection-sampling 概率诊断，因此为空；线性模式保持原定义。

## 验证

单元测试覆盖：

- 树 attention mask 的上下文和祖先可见性；
- 全局 top-N 打包的父节点前置与前缀闭合；
- draft-to-target 词表映射；
- RNN Markov 状态及 `sample_from_anchor=false` 的保留 slot 重放；
- target argmax 接受路径、bonus token、KV cache 和 hidden-state gather；
- 原线性 DSpark、Correction、candidate selector 和统计路径的回归行为。
