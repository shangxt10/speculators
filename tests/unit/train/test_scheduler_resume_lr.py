"""The first optimizer update after reload uses the saved learning rate."""

import copy

import pytest
import torch
from transformers import get_cosine_schedule_with_warmup

from speculators.train.checkpointer import SingleGPUCheckpointer


@pytest.mark.parametrize("multiple", [False, True])
def test_resumed_first_update_matches_uninterrupted(tmp_path, multiple):
    count = 2 if multiple else 1
    params = [torch.nn.Parameter(torch.ones(2)) for _ in range(count)]
    optimizers = [
        torch.optim.AdamW([p], lr=(i + 1) * 1e-4) for i, p in enumerate(params)
    ]
    schedulers = [get_cosine_schedule_with_warmup(o, 4, 100) for o in optimizers]
    for _ in range(50):
        for param, opt, sched in zip(params, optimizers, schedulers, strict=True):
            param.grad = torch.ones_like(param)
            opt.step()
            sched.step()
    folder = tmp_path / "0"
    folder.mkdir()
    saved = [copy.deepcopy(s.state_dict()) for s in schedulers]
    torch.save(saved if multiple else saved[0], folder / "scheduler_state_dict.pt")
    clones = [torch.nn.Parameter(p.detach().clone()) for p in params]
    resumed = [torch.optim.AdamW([p], lr=(i + 1) * 1e-4) for i, p in enumerate(clones)]
    restored_schedulers = []
    for old, new in zip(optimizers, resumed, strict=True):
        new.load_state_dict(copy.deepcopy(old.state_dict()))
        restored_schedulers.append(
            get_cosine_schedule_with_warmup(new, 4, 100, last_epoch=0)
        )
    checkpointer = SingleGPUCheckpointer(tmp_path)
    checkpointer.load_scheduler_state_dict(
        restored_schedulers if multiple else restored_schedulers[0]
    )
    for before, after in zip(optimizers, resumed, strict=True):
        assert after.param_groups[0]["lr"] == before.param_groups[0]["lr"]
    for p, q, old, new in zip(params, clones, optimizers, resumed, strict=True):
        p.grad = torch.ones_like(p)
        q.grad = torch.ones_like(q)
        old.step()
        new.step()
        torch.testing.assert_close(p, q, rtol=0, atol=0)
