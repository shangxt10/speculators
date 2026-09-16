"""HCCL process-group lifecycle for one pure expert-parallel group."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExpertParallelContext:
    rank: int
    local_rank: int
    world_size: int
    device: Any
    device_mesh: Any

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        import torch.distributed as dist

        dist.barrier()

    def synchronize(self) -> None:
        import torch

        torch.npu.synchronize(self.device)


def initialize_expert_parallel(expected_world_size: int) -> ExpertParallelContext:
    """Initialize one HCCL process per NPU and one one-dimensional EP mesh."""
    try:
        import torch
        import torch.distributed as dist
        import torch_npu  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "Pure PyTorch Ascend EP requires torch, torch_npu, and HCCL."
        ) from error

    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(
            "Launch this entry point with torchrun. Missing environment variables: "
            + ", ".join(missing)
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != expected_world_size:
        raise ValueError(
            f"torchrun WORLD_SIZE={world_size}, but --expert-parallel-size="
            f"{expected_world_size}. This implementation uses one EP group and no "
            "TP/DP/PP groups."
        )

    torch.npu.set_device(local_rank)
    device = torch.device("npu", local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="hccl",
            rank=rank,
            world_size=world_size,
        )

    from torch.distributed.device_mesh import init_device_mesh

    device_mesh = init_device_mesh(
        "npu",
        (world_size,),
        mesh_dim_names=("ep",),
    )
    return ExpertParallelContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        device_mesh=device_mesh,
    )


def destroy_expert_parallel(context: ExpertParallelContext) -> None:
    import torch.distributed as dist

    if not dist.is_initialized():
        return
    try:
        context.barrier()
    finally:
        dist.destroy_process_group()

