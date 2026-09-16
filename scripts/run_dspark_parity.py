#!/usr/bin/env python3
"""One-command DSpark parity run against extraction and DSpark services."""

import argparse
import subprocess
import sys
from pathlib import Path

import openai

from debug_dspark_prompt import DEFAULT_PROMPT, evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--verifier", required=True)
    parser.add_argument("--hidden-states-path", type=Path, required=True)
    parser.add_argument("--extract-endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--dspark-endpoint", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--speculators-trace", type=Path, required=True)
    parser.add_argument("--vllm-trace", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    existing = list(args.speculators_trace.glob("*.pt"))
    existing.extend(args.vllm_trace.glob("rank_*/step_*/*.pt"))
    if existing:
        raise FileExistsError(
            "Trace output is not empty. Use fresh --speculators-trace and "
            "--vllm-trace directories to avoid comparing stale files."
        )

    evaluate(
        argparse.Namespace(
            draft_model=args.draft_model,
            verifier=args.verifier,
            extract_endpoint=args.extract_endpoint,
            hidden_states_path=args.hidden_states_path,
            output_dir=args.speculators_trace,
            device=args.device,
            prompt=args.prompt,
            trust_remote_code=args.trust_remote_code,
        )
    )

    args.vllm_trace.mkdir(parents=True, exist_ok=True)
    arm_path = args.vllm_trace / "ARM"
    arm_path.touch()
    try:
        client = openai.OpenAI(
            base_url=args.dspark_endpoint,
            api_key="EMPTY",
            max_retries=0,
        )
        model_id = client.models.list().data[0].id
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": args.prompt}],
            temperature=0,
            max_tokens=8,
        )
        print(f"vLLM response: {response.choices[0].message.content!r}")
    finally:
        arm_path.unlink(missing_ok=True)

    compare_script = Path(__file__).with_name("compare_dspark_traces.py")
    subprocess.run(
        [
            sys.executable,
            str(compare_script),
            "--speculators-trace",
            str(args.speculators_trace),
            "--vllm-trace",
            str(args.vllm_trace),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
