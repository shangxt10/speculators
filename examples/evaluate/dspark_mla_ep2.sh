#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

unset HCCL_BUFFSIZE
unset HCCL_OP_EXPANSION_MODE

# 必须使用包含本次 MLA 离线评测适配的代码目录。
CODE_ROOT="${CODE_ROOT:-/hpfs/huawei-2607/xiaoyf/speculators_mla/speculators}"
export PYTHONPATH="${CODE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

VERIFIER="/hpfs/nlp/yangzhihao/ckpts/TeleChat4-29B/29b_12.05T_wodsa_id20_s3_256k_8_260903_1000_final_hf"
DRAFT="/hpfs/huawei-2607/shangxt/Telecom-29B/open_perfect_80w_no_thinking_full_vocab_1014_32_96_b7_mla_new/checkpoints/checkpoint_best"
DATA_ROOT="/hpfs/huawei-2607/xiaoyf/speculators_mla/speculators"

# MLA 结构从 DRAFT/config.json 自动恢复；sdpa 是计算后端。
# 1：不启用树投机；>=2：启用树投机，例如 TREE_TOP_K=4。
TREE_TOP_K="${TREE_TOP_K:-1}"
TREE_MAX_NODES="${TREE_MAX_NODES:-64}"
if [[ "${TREE_TOP_K}" == "1" ]]; then
    OUTPUT_ROOT="${OUTPUT_ROOT:-./eval_results/bs7_mla_notree}"
else
    OUTPUT_ROOT="${OUTPUT_ROOT:-./eval_results/bs7_mla_tree_k${TREE_TOP_K}}"
fi

mkdir -p "${OUTPUT_ROOT}/logs"
JOB_COUNT=0

launch_ep2() {
    local devices="$1"
    local datasets="$2"
    local name="$3"

    echo "Launching ${name}: devices=${devices}, datasets=${datasets}, tree_top_k=${TREE_TOP_K}"

    ASCEND_RT_VISIBLE_DEVICES="${devices}" \
    nohup torchrun \
      --standalone \
      --nnodes=1 \
      --nproc-per-node=2 \
      "${CODE_ROOT}/xingchen4_eval_ep/evaluate_ep.py" \
      --verifier-model "${VERIFIER}" \
      --draft-model "${DRAFT}" \
      --datasets-root "${DATA_ROOT}" \
      --datasets "${datasets}" \
      --max-samples 100 \
      --output-dir "${OUTPUT_ROOT}/${name}" \
      --expert-parallel-size 2 \
      --max-new-tokens 1024 \
      --temperature 0 \
      --tree-top-k "${TREE_TOP_K}" \
      --tree-max-nodes "${TREE_MAX_NODES}" \
      --dtype bfloat16 \
      --draft-attn-impl sdpa \
      --enable-thinking false \
      --raw-prompt-mode auto \
      --skip-artifacts \
      --log-every 1 \
      --trust-remote-code \
      --local-files-only \
      > "${OUTPUT_ROOT}/logs/${name}.log" 2>&1 < /dev/null &

    echo $! > "${OUTPUT_ROOT}/logs/${name}.pid"
    JOB_COUNT=$((JOB_COUNT + 1))
}

launch_ep2 "0,1"   "gsm8k"                "gsm8k"

#launch_ep2 "2,3"   "math500"              "math500"
#launch_ep2 "4,5"   "humaneval"             "humaneval"
#launch_ep2 "6,7"   "mbpp"                  "mbpp"
#launch_ep2 "8,9"   "livecodebench"         "livecodebench"
#launch_ep2 "10,11" "alpaca"                "alpaca"
#launch_ep2 "12,13" "arena-hard-v2"         "arena-hard-v2"
#launch_ep2 "14,15" "aime25,mt-bench"       "aime25_mt-bench"

echo "Launched ${JOB_COUNT} EP2 job(s). Logs: ${OUTPUT_ROOT}/logs"
