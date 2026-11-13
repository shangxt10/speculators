#!/usr/bin/env python3
"""Inspect a local verifier config without allocating model weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from ep_plan import resolve_ep_plan, validate_ep_plan  # noqa: E402
from modulelist_ep import detect_expert_layout  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expert-parallel-size", type=int, required=True)
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
    args = parser.parse_args()

    from transformers import AutoConfig
    from transformers.integrations.tensor_parallel import ALL_PARALLEL_STYLES

    config = AutoConfig.from_pretrained(
        str(args.model),
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    plan = resolve_ep_plan(config)
    expert_layout = detect_expert_layout(args.model)
    validate_ep_plan(
        plan,
        ep_size=args.expert_parallel_size,
        available_styles=(
            set(ALL_PARALLEL_STYLES) if expert_layout == "fused" else None
        ),
    )
    implementation = (
        "native_grouped_gemm"
        if expert_layout == "fused"
        else "custom_modulelist"
    )
    print(
        json.dumps(
            plan.to_dict(
                args.expert_parallel_size,
                implementation=implementation,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
