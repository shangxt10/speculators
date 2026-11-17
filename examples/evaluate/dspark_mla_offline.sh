#!/usr/bin/env bash
# Use a trained DSpark MLA checkpoint. Layer types are restored from config.json.
set -euo pipefail

: "${MODEL:?Set MODEL to the target model directory}"
: "${DRAFT_MODEL:?Set DRAFT_MODEL to the trained draft checkpoint directory}"
: "${DATASETS_ROOT:?Set DATASETS_ROOT to the evaluation JSONL directory}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
EVAL_MODE="${EVAL_MODE:-tree}"       # token | tree
EVAL_BACKEND="${EVAL_BACKEND:-single}" # single | ep
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/eval_results/dspark_mla_${EVAL_BACKEND}_${EVAL_MODE}}"

ARGS=(
    --verifier-model "$MODEL"
    --draft-model "$DRAFT_MODEL"
    --datasets-root "$DATASETS_ROOT"
    --output-dir "$OUTPUT_DIR"
    --dtype "${DTYPE:-bfloat16}"
    --draft-attn-impl sdpa
    --temperature 0
    --max-new-tokens "${MAX_NEW_TOKENS:-512}"
    --measure-base-speedup
    --trust-remote-code
)
if [[ -n "${DATASETS:-}" ]]; then
    ARGS+=(--datasets "$DATASETS")
fi
case "$EVAL_MODE" in
    token) ARGS+=(--tree-top-k 1) ;;
    tree) ARGS+=(--tree-top-k "${TREE_TOP_K:-4}" --tree-max-nodes "${TREE_MAX_NODES:-32}") ;;
    *) echo "EVAL_MODE must be token or tree" >&2; exit 1 ;;
esac

case "$EVAL_BACKEND" in
    single)
        exec "${PYTHON:-python}" "$REPO_ROOT/scripts/evaluate/dspark_offline_eval.py" \
            --device "${DEVICE:-npu:0}" "${ARGS[@]}" "$@"
        ;;
    ep)
        exec "${PYTHON:-python}" -m torch.distributed.run \
            --standalone --nnodes=1 --nproc-per-node="${EP_SIZE:-16}" \
            "$REPO_ROOT/xingchen4_eval_ep/evaluate_ep.py" \
            --expert-parallel-size "${EP_SIZE:-16}" "${ARGS[@]}" "$@"
        ;;
    *) echo "EVAL_BACKEND must be single or ep" >&2; exit 1 ;;
esac
