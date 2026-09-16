#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import torch


def _load(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=True)


def _normalize(value: torch.Tensor) -> torch.Tensor:
    while value.ndim > 1 and value.shape[0] == 1:
        value = value.squeeze(0)
    return value


def _metrics(left: torch.Tensor, right: torch.Tensor) -> tuple[str, ...]:
    left = _normalize(left)
    right = _normalize(right)
    if left.shape != right.shape:
        return (
            str(list(left.shape)),
            str(list(right.shape)),
            "shape mismatch",
            "",
            "",
        )
    if not left.is_floating_point() and not right.is_floating_point():
        equal = torch.equal(left, right)
        return (
            str(list(left.shape)),
            str(list(right.shape)),
            str(equal),
            "",
            "",
        )
    left_f = left.float().reshape(-1)
    right_f = right.float().reshape(-1)
    delta = (left_f - right_f).abs()
    cosine = torch.nn.functional.cosine_similarity(left_f, right_f, dim=0)
    return (
        str(list(left.shape)),
        str(list(right.shape)),
        f"{float(delta.max()):.6g}",
        f"{float(delta.mean()):.6g}",
        f"{float(cosine):.9f}" if math.isfinite(float(cosine)) else "nan",
    )


def _resolve_vllm_step(path: Path) -> Path:
    if (path / "base_logits.pt").exists():
        return path
    candidates = sorted(path.glob("rank_*/step_*"))
    if not candidates:
        raise FileNotFoundError(f"No rank_*/step_* trace found below {path}")
    rank_zero = [
        candidate for candidate in candidates if candidate.parent.name == "rank_0000"
    ]
    return rank_zero[0] if rank_zero else candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Speculators and vLLM DSpark traces"
    )
    parser.add_argument("--speculators-trace", type=Path, required=True)
    parser.add_argument("--vllm-trace", type=Path, required=True)
    args = parser.parse_args()

    spec_dir = args.speculators_trace
    vllm_dir = _resolve_vllm_step(args.vllm_trace)
    spec_names = {path.stem for path in spec_dir.glob("*.pt")}
    vllm_names = {path.stem for path in vllm_dir.glob("*.pt")}
    common = sorted(spec_names & vllm_names)
    if not common:
        raise RuntimeError("The two trace directories have no common tensor names")

    print(f"Speculators: {spec_dir}")
    print(f"vLLM:       {vllm_dir}")
    print("field\tspec_shape\tvllm_shape\tmax_abs/exact\tmean_abs\tcosine")
    for name in common:
        row = _metrics(_load(spec_dir / f"{name}.pt"), _load(vllm_dir / f"{name}.pt"))
        print("\t".join((name, *row)))

    missing_spec = sorted(vllm_names - spec_names)
    missing_vllm = sorted(spec_names - vllm_names)
    if missing_spec:
        print(f"Only in vLLM: {', '.join(missing_spec)}")
    if missing_vllm:
        print(f"Only in Speculators: {', '.join(missing_vllm)}")


if __name__ == "__main__":
    main()
