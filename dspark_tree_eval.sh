#!/usr/bin/env bash

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

unset HCCL_BUFFSIZE
unset HCCL_OP_EXPANSION_MODE

VERIFIER="/hpfs/nlp/yangzhihao/ckpts/TeleChat4-29B/29b_12.05T_wodsa_id20_s3_256k_8_260903_1000_final_hf"
DRAFT="/hpfs/huawei-2607/shangxt/Telecom-29B/open_perfect_80w_no_thinking_full_vocab_1014_32_96_rnn/checkpoints/checkpoint_best"
DATA_ROOT="/hpfs/huawei-2607/shangxt/evaldata"
OUTPUT_ROOT="./eval_results/xingchen4_pytorch_ep2_16npu_140w_rnn_rope_tree"

mkdir -p "${OUTPUT_ROOT}/logs"

launch_ep2() {
local devices="$1"
local datasets="$2"
local name="$3"

echo "Launching ${name}: devices=${devices}, datasets=${datasets}"

ASCEND_RT_VISIBLE_DEVICES="${devices}" \
nohup torchrun \
  --standalone \
  --nproc-per-node=2 \
  ./xingchen4_eval_ep_tree/evaluate_ep.py \
  --verifier-model "${VERIFIER}" \
  --draft-model "${DRAFT}" \
  --datasets-root "${DATA_ROOT}" \
  --datasets "${datasets}" \
  --output-dir "${OUTPUT_ROOT}/${name}" \
  --expert-parallel-size 2 \
  --max-new-tokens 2048 \
  --temperature 0 \
  --tree-top-k 4 \
  --tree-max-nodes 32 \
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

}

launch_ep2 "0,1"   "gsm8k"                "gsm8k"
launch_ep2 "2,3"   "math500"              "math500"
launch_ep2 "4,5"   "humaneval"             "humaneval"
launch_ep2 "6,7"   "mbpp"                  "mbpp"
launch_ep2 "8,9"   "livecodebench"         "livecodebench"
launch_ep2 "10,11" "alpaca"                "alpaca"
launch_ep2 "12,13" "arena-hard-v2"         "arena-hard-v2"
launch_ep2 "14,15" "aime25,mt-bench"       "aime25_mt-bench"

echo "All eight EP2 jobs launched."
