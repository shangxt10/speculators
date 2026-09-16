#!/usr/bin/env python3
"""Pure PyTorch/HCCL expert-parallel DSpark offline evaluator."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path
from types import ModuleType


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from model_loader import (  # noqa: E402
    LocalOutputTargetAdapter,
    load_expert_parallel_target,
)
from npu_format import install_draft_attention_cat_compatibility  # noqa: E402
from parallel_state import (  # noqa: E402
    ExpertParallelContext,
    destroy_expert_parallel,
    initialize_expert_parallel,
)


LOGGER = logging.getLogger("dspark_pytorch_ep_eval")


def _load_legacy_evaluator() -> ModuleType:
    path = MODULE_DIR.parent / "dspark_offline_eval.py"
    spec = importlib.util.spec_from_file_location(
        "dspark_offline_eval_readonly_for_ep",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load existing evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _configure_logging(context: ExpertParallelContext) -> None:
    level = logging.INFO if context.is_primary else logging.WARNING
    logging.basicConfig(
        level=level,
        format="[%(levelname)s][rank %(process)d] %(message)s",
    )


def _write_run_config(
    args: argparse.Namespace,
    context: ExpertParallelContext,
) -> None:
    if not context.is_primary:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    values = vars(args).copy()
    for key, value in values.items():
        if isinstance(value, Path):
            values[key] = str(value)
    values.update(
        {
            "parallelism": "expert_parallel_only",
            "expert_parallel_enabled": True,
            "tensor_parallel_enabled": False,
            "world_size": context.world_size,
        }
    )
    with (args.output_dir / "pytorch_ep_run_config.json").open(
        "w",
        encoding="utf-8",
    ) as file_obj:
        json.dump(values, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")


def _load_draft(
    legacy: ModuleType,
    args: argparse.Namespace,
    context: ExpertParallelContext,
):
    from speculators.models.dspark.core import DSparkDraftModel

    config = DSparkDraftModel.config_class.from_pretrained(args.draft_model)
    config.transformer_layer_config._attn_implementation = args.draft_attn_impl

    verifier = getattr(getattr(config, "speculators_config", None), "verifier", None)
    if verifier is not None:
        verifier.name_or_path = str(args.verifier_model)

    sample_from_anchor = legacy._parse_bool_override(args.sample_from_anchor)
    if sample_from_anchor is not None:
        config.sample_from_anchor = sample_from_anchor

    d2t, t2d = legacy._load_vocab_mapping_tensors(
        draft_model_path=args.draft_model,
        d2t_path=args.d2t_path,
        t2d_path=args.t2d_path,
    )
    model = DSparkDraftModel.from_pretrained(
        args.draft_model,
        config=config,
        d2t=d2t,
        t2d=t2d,
    )
    model = model.to(context.device).eval()
    patch_count = install_draft_attention_cat_compatibility(model)
    if context.is_primary:
        LOGGER.info(
            "Installed NPU-safe K/V concatenation on %d draft attention layers",
            patch_count,
        )
    legacy._ensure_loaded_vocab_mappings(model, args)
    return model


def _prepare_legacy_args(
    args: argparse.Namespace,
    context: ExpertParallelContext,
) -> argparse.Namespace:
    args.device = str(context.device)
    args.ascend_devices = None
    args.worker_shard_index = None
    args.worker_num_shards = 1
    args.no_progress = args.no_progress or not context.is_primary
    if not context.is_primary:
        args.skip_artifacts = True
    return args


def run(args: argparse.Namespace) -> None:
    if args.temperature != 0:
        raise ValueError(
            "Pure EP evaluation currently requires --temperature 0 so every "
            "rank follows identical speculative-decoding control flow."
        )

    context = initialize_expert_parallel(args.expert_parallel_size)
    _configure_logging(context)
    _write_run_config(args, context)
    legacy = _load_legacy_evaluator()

    import torch
    from transformers import AutoTokenizer, DynamicCache

    legacy.torch = torch
    legacy.DynamicCache = DynamicCache
    torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)
    args = _prepare_legacy_args(args, context)

    try:
        draft_model = _load_draft(legacy, args, context)
        target_layer_ids = list(draft_model.target_layer_ids)
        target_model, plan = load_expert_parallel_target(
            model_path=args.verifier_model,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
            context=context,
            plan_output_path=args.output_dir / "resolved_ep_plan.json",
        )
        target_model = LocalOutputTargetAdapter(
            target_model,
            target_layer_ids,
            expected_feature_width=int(draft_model.fc.in_features),
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer or args.verifier_model,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )

        if context.is_primary:
            LOGGER.info(
                "Loaded pure EP verifier (%s): %d experts, %d experts/rank, "
                "%d declared rules",
                getattr(
                    target_model.model,
                    "_dspark_ep_implementation",
                    "unknown",
                ),
                plan.num_experts,
                plan.num_experts // context.world_size,
                len(plan.rules),
            )
            LOGGER.info(
                "Dense layers, attention, shared experts, lm_head, and draft are "
                "replicated; routed experts alone are sharded"
            )
            if target_model.hidden_stream_collapser_name is not None:
                LOGGER.info(
                    "Xingchen mHC hidden states will be collapsed with verifier "
                    "module %s before DSpark projection",
                    target_model.hidden_stream_collapser_name,
                )

        runner = legacy.DSparkOfflineRunner(
            target_model,
            draft_model,
            tokenizer,
            args,
        )
        base_runner = (
            legacy.BaseModelOfflineRunner(target_model, tokenizer, args)
            if args.measure_base_speedup
            else None
        )
        stop_token_ids = legacy.resolve_stop_token_ids(target_model, tokenizer)
        dataset_paths = legacy._discover_datasets(
            args.datasets_root,
            legacy._split_csv(args.datasets) or None,
        )
        rows = []
        artifacts_by_dataset = {}

        for dataset_path in dataset_paths:
            context.barrier()
            row, artifacts = legacy._evaluate_dataset(
                path=dataset_path,
                runner=runner,
                base_runner=base_runner,
                args=args,
                stop_token_ids=stop_token_ids,
            )
            context.synchronize()
            context.barrier()
            if context.is_primary:
                rows.append(row)
                if not args.skip_artifacts:
                    artifacts_by_dataset[dataset_path.stem] = artifacts
                legacy._write_outputs(
                    args.output_dir,
                    rows,
                    artifacts_by_dataset,
                )
                LOGGER.info(
                    "[%s] acceptance_length=%.4f",
                    dataset_path.stem,
                    row["acceptance_length"],
                )
    finally:
        destroy_expert_parallel(context)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pure PyTorch expert-parallel evaluation for "
            "Xingchen4 MoE + DSpark. No vLLM and no tensor parallelism."
        )
    )
    parser.add_argument("--verifier-model", type=Path, required=True)
    parser.add_argument("--draft-model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expert-parallel-size", type=int, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=980406)
    parser.add_argument(
        "--enable-thinking",
        choices=("false", "true", "default"),
        default="false",
    )
    parser.add_argument(
        "--raw-prompt-mode",
        choices=("auto", "chat_template", "raw"),
        default="auto",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--draft-attn-impl",
        choices=("sdpa", "eager"),
        default="sdpa",
    )
    parser.add_argument("--d2t-path", type=Path, default=None)
    parser.add_argument("--t2d-path", type=Path, default=None)
    parser.add_argument(
        "--sample-from-anchor",
        choices=("true", "false"),
        default=None,
    )
    parser.add_argument("--measure-base-speedup", action="store_true")
    parser.add_argument("--throughput-warmup-samples", type=int, default=1)
    parser.add_argument("--skip-artifacts", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    run(parse_args())


if __name__ == "__main__":
    main()
