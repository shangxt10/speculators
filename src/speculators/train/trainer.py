import json
import logging
import time
import warnings
from pathlib import Path
from typing import Literal, NamedTuple

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
)
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from speculators.model import SpeculatorModel
from speculators.train.checkpointer import (
    BaseCheckpointer,
    DistributedCheckpointer,
    SingleGPUCheckpointer,
)
from speculators.train.data import HS_PROFILE_KEY
from speculators.train.distributed import (
    apply_fully_sharded,
    get_local_rank,
    get_rank,
    is_distributed,
)
from speculators.train.graceful_shutdown import with_graceful_shutdown
from speculators.train.optimizers import build_optimizers
from speculators.train.utils import normalize_counted_metrics

root_logger = logging.getLogger("speculators")
metric_logger = logging.getLogger("speculators.metrics")


def _synchronize_device() -> None:
    """Best-effort device sync for step profiling (CUDA or NPU).

    Never raises: profiling must not abort training if the device sync path is
    missing or broken on a given accelerator backend.
    """
    try:
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        return


class _StepTimer:
    # Each mark()/now() forces a device synchronize to capture true accelerator
    # time. This serialises the device pipeline, so profiled steps are slower;
    # keep log_freq > 1 in perf-sensitive runs.
    _PROFILE_MARKS = ("fetch", "fwd", "bwd", "opt")

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._marks: dict[str, float] = {}

    def reset(self, enabled: bool) -> None:
        self.enabled = enabled
        self._marks.clear()

    def mark(self, name: str) -> None:
        if not self.enabled:
            return
        _synchronize_device()
        self._marks[name] = time.perf_counter()

    def mark_value(self, name: str, value: float) -> None:
        if self.enabled:
            self._marks[name] = value

    def now(self) -> float | None:
        if not self.enabled:
            return None
        _synchronize_device()
        return time.perf_counter()

    def profile(self, num_tokens: int) -> dict[str, float] | None:
        if not self.enabled:
            return None
        m = self._marks
        if any(name not in m for name in self._PROFILE_MARKS):
            return None
        has_start = "start" in m
        has_data_ready = "data_ready" in m
        fwd_started = m.get("pre_fwd", m["fetch"])
        pre_fwd_sync_ms = (fwd_started - m["fetch"]) * 1000
        fwd_ms = (m["fwd"] - fwd_started) * 1000
        bwd_ms = (m["bwd"] - m["fwd"]) * 1000
        opt_ms = (m["opt"] - m["bwd"]) * 1000
        fetch_ms = (m["fetch"] - m["start"]) * 1000 if has_start else 0.0
        loader_wait_ms = (
            (m["data_ready"] - m["start"]) * 1000
            if has_start and has_data_ready
            else fetch_ms
        )
        h2d_ms = (
            (m["fetch"] - m["data_ready"]) * 1000 if has_data_ready else 0.0
        )
        step_ms = (m["opt"] - m["start"]) * 1000 if has_start else 0.0
        tokens_per_s = num_tokens / (step_ms / 1000) if step_ms > 0 else 0.0
        fetch_frac = fetch_ms / step_ms if step_ms > 0 else 0.0
        return {
            "loader_wait_ms": loader_wait_ms,
            "h2d_ms": h2d_ms,
            "fetch_ms": fetch_ms,
            "pre_fwd_sync_ms": pre_fwd_sync_ms,
            "fwd_ms": fwd_ms,
            "bwd_ms": bwd_ms,
            "opt_ms": opt_ms,
            "step_ms": step_ms,
            "tokens_per_s": tokens_per_s,
            "fetch_frac": fetch_frac,
        }


warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
MIN_STEP_PCT = 0.25


class TrainerConfig(NamedTuple):
    lr: float
    num_epochs: int
    save_path: str
    resume_from_checkpoint: bool = False
    train_call_kwargs: dict | None = None
    val_call_kwargs: dict | None = None
    optimizer: Literal["adamw", "muon"] = "adamw"
    weight_decay: float = 0.01
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.1
    muon_ns_steps: int = 5
    muon_adjust_lr_fn: str = "match_rms_adamw"
    scheduler_type: Literal["linear", "cosine", "none"] = "linear"
    scheduler_warmup_steps: int | None = None
    scheduler_warmup_ratio: float | None = None
    scheduler_total_steps: int | None = None
    scheduler_num_cosine_cycles: float = 0.5
    checkpoint_freq: float = 1
    save_best: bool = False
    hidden_states_dtype: torch.dtype = torch.bfloat16
    log_freq: int = 1
    fsdp_shard: bool = False
    profile_pipeline: bool = False
    profile_sync_ranks: bool = False
    profile_summary_freq: int = 10


def _resolve_scheduler_steps(
    config: TrainerConfig,
    train_loader_len: int,
) -> tuple[int, int]:
    """Resolve ``(warmup_steps, total_steps)`` for the LR scheduler.

    Explicit ``scheduler_warmup_steps`` wins; otherwise ``scheduler_warmup_ratio``
    (a fraction of total steps, validated to ``[0, 1]``) is used; otherwise the
    default of 1% of the resolved total steps. ``scheduler_total_steps`` defaults
    to ``num_epochs * train_loader_len``.
    """
    default_total_steps = config.num_epochs * train_loader_len
    scheduler_total_steps = (
        config.scheduler_total_steps
        if config.scheduler_total_steps is not None
        else default_total_steps
    )

    if config.scheduler_warmup_steps is not None:
        scheduler_warmup_steps = config.scheduler_warmup_steps
        if config.scheduler_warmup_ratio is not None:
            warnings.warn(
                "Both scheduler_warmup_steps and scheduler_warmup_ratio are set; "
                "using scheduler_warmup_steps.",
                stacklevel=2,
            )
    elif config.scheduler_warmup_ratio is not None:
        if not 0 <= config.scheduler_warmup_ratio <= 1:
            raise ValueError("scheduler_warmup_ratio must be between 0 and 1.")
        scheduler_warmup_steps = int(
            scheduler_total_steps * config.scheduler_warmup_ratio
        )
    else:
        scheduler_warmup_steps = scheduler_total_steps // 100

    return scheduler_warmup_steps, scheduler_total_steps


class Trainer:
    def __init__(
        self,
        model: SpeculatorModel,
        config: TrainerConfig,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ):
        self.model = model
        self.config = config
        self.local_rank = get_local_rank()
        self.rank = get_rank()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.is_distributed = is_distributed()
        self.resume_from_checkpoint = config.resume_from_checkpoint
        acc = torch.accelerator.current_accelerator()
        self.device_type = acc.type if acc is not None else "cuda"
        checkpointer_class: type[BaseCheckpointer] = (
            DistributedCheckpointer
            if self.is_distributed and config.fsdp_shard
            else SingleGPUCheckpointer
        )
        self.checkpointer: BaseCheckpointer = checkpointer_class(self.config.save_path)

        self.setup_trainer()
        self.setup_model()
        self.setup_optimizer()

    def _training_state_path(self, epoch: int) -> Path:
        return self.checkpointer.path / str(epoch) / "training_state.json"

    def _save_training_state(self, epoch: int, local_step: int) -> None:
        if not self.is_distributed or dist.get_rank() == 0:
            state = {
                "epoch": epoch,
                "local_step": local_step,
                "global_step": self.global_step,
            }
            p = self._training_state_path(epoch)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(state))

    def _load_training_state(self) -> dict:
        epoch = self.checkpointer.previous_epoch
        p = self._training_state_path(epoch)
        if p.exists():
            try:
                return json.loads(p.read_text())
            except json.JSONDecodeError as e:
                root_logger.warning(f"Failed to decode training state {p}: {e}")
            except (FileNotFoundError, PermissionError, OSError) as e:
                root_logger.warning(f"Failed to read training state {p}: {e}")
        return {}

    def setup_trainer(self):
        if self.checkpointer.previous_epoch != -1:
            root_logger.info(f"Found checkpoint at {self.checkpointer.prev_path}.")
            self.current_epoch = self.checkpointer.previous_epoch + 1
            if self.resume_from_checkpoint:
                # Check if this was a mid-epoch checkpoint — if so, resume
                # from within that epoch rather than jumping to the next one.
                state = self._load_training_state()
                is_mid_epoch = (
                    state
                    and state.get("epoch") == self.checkpointer.previous_epoch
                    and state.get("local_step", 0) > 0  # 0 means end-of-epoch
                )
                if is_mid_epoch:
                    # Resume within the same epoch from the exact step.
                    self.current_epoch = state["epoch"]
                    self._resume_local_step = state["local_step"]
                    self._resume_global_step = state.get("global_step", 0)
                    root_logger.info(
                        f"Resuming mid-epoch from epoch={self.current_epoch} "
                        f"local_step={self._resume_local_step} "
                        f"global_step={self._resume_global_step}."
                    )
                else:
                    # End-of-epoch or no state — advance to next epoch.
                    self._resume_local_step = 0
                    resume_global = state.get("global_step", 0) if state else 0
                    self._resume_global_step = resume_global
                    root_logger.info(
                        f"Resuming training on epoch {self.current_epoch}."
                    )
            else:
                root_logger.warning(
                    "`resume_from_checkpoint` is False, starting "
                    "training from scratch. This will overwrite the "
                    f"existing checkpoints in {self.checkpointer.path}."
                )
                self.current_epoch = 0
                self._resume_local_step = 0
                self._resume_global_step = 0
        else:
            root_logger.info(
                "No previous training checkpoint found in "
                f"'{self.checkpointer.path}'. Starting fresh training run."
            )
            self.current_epoch = 0
            self._resume_local_step = 0
            self._resume_global_step = 0
        self.global_step = self._resume_global_step
        self.best_val_loss = float("inf")

        if self.resume_from_checkpoint and self.checkpointer.previous_epoch != -1:
            saved = self.checkpointer.load_best_val_loss()
            if saved is not None:
                self.best_val_loss = saved
                root_logger.info(
                    f"Restored best_val_loss={self.best_val_loss:.6f} from checkpoint"
                )

    def setup_model(self):
        # Verify model is compatible with training infrastructure
        SpeculatorModel.verify_training_compatible(self.model)

        load_checkpoint = (
            self.resume_from_checkpoint and self.checkpointer.previous_epoch != -1
        )

        if not self.is_distributed:
            self.model.to(self.local_rank)  # type: ignore[arg-type]
            if load_checkpoint:
                self.checkpointer.load_model_state_dict(self.model)
            return

        if self.config.fsdp_shard:
            self._setup_model_fsdp(load_checkpoint)
        else:
            self._setup_model_ddp(load_checkpoint)

    def _setup_model_fsdp(self, load_checkpoint: bool):
        # Capture full state dict on rank 0 before FSDP sharding
        full_state_dict = {}
        if not load_checkpoint and dist.get_rank() == 0:
            full_state_dict = self.model.state_dict()

        apply_fully_sharded(self.model, param_dtype=self.config.hidden_states_dtype)

        if load_checkpoint:
            self.checkpointer.load_model_state_dict(self.model)
        else:
            # Broadcast full state dict from rank 0 to all ranks
            set_model_state_dict(
                self.model,
                full_state_dict,
                options=StateDictOptions(
                    full_state_dict=True,
                    broadcast_from_rank0=True,
                    strict=False,
                ),
            )
            del full_state_dict
            dist.barrier()

    def _setup_model_ddp(self, load_checkpoint: bool):
        self.model.to(self.local_rank)  # type: ignore[arg-type]

        if load_checkpoint:
            if dist.get_rank() == 0:
                self.checkpointer.load_model_state_dict(self.model)
        else:
            # Fresh init: broadcast rank 0's random initialization to all ranks
            for param in self.model.parameters():
                dist.broadcast(param.data, src=0)
            dist.barrier()

        # DDP constructor broadcasts rank 0's params to all ranks
        self.model = DistributedDataParallel(self.model)  # type: ignore[assignment]

    def setup_optimizer(self):
        # Setup optimizer(s). The "muon" option returns two optimizers (Muon for the
        # 2D weight matrices, AdamW for everything else); "adamw" returns a single one.
        self.optimizers = build_optimizers(self.model, self.config)
        last_epoch = -1
        if self.resume_from_checkpoint and self.checkpointer.previous_epoch != -1:
            self.checkpointer.load_optimizer_state_dict(self.model, self.optimizers)
            last_epoch = self.checkpointer.previous_epoch

        # Setup scheduler(s) — one per optimizer so each optimizer's base LR (e.g.
        # Muon's higher LR vs AdamW's) is warmed up / decayed independently.
        if self.config.scheduler_type == "none":
            self.schedulers: list[torch.optim.lr_scheduler.LRScheduler] = []
            return

        scheduler_warmup_steps, scheduler_total_steps = _resolve_scheduler_steps(
            self.config, len(self.train_loader)
        )

        def make_scheduler(opt: torch.optim.Optimizer):
            if self.config.scheduler_type == "linear":
                return get_linear_schedule_with_warmup(
                    opt,
                    num_warmup_steps=scheduler_warmup_steps,
                    num_training_steps=scheduler_total_steps,
                    last_epoch=last_epoch,
                )
            return get_cosine_schedule_with_warmup(
                opt,
                num_warmup_steps=scheduler_warmup_steps,
                num_training_steps=scheduler_total_steps,
                num_cycles=self.config.scheduler_num_cosine_cycles,
                last_epoch=last_epoch,
            )

        self.schedulers = [make_scheduler(opt) for opt in self.optimizers]

        if self.resume_from_checkpoint and self.checkpointer.previous_epoch != -1:
            self.checkpointer.load_scheduler_state_dict(self.schedulers)

    def _optimizers_zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def _optimizers_step(self):
        for opt in self.optimizers:
            opt.step()

    def _schedulers_step(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def _prepare_resume_skip(self, epoch: int) -> int:
        """Prepare fast-skip state for mid-epoch resume and return skipped steps."""
        skip_steps = 0
        if epoch == getattr(self, "current_epoch", epoch):
            skip_steps = getattr(self, "_resume_local_step", 0)
            # Only skip once — clear after use.
            self._resume_local_step = 0

        # Fast-skip: slice the sampler's pre-generated batch list so we never
        # call __getitem__ (and thus never call vLLM) for skipped batches.
        sampler = self.train_loader.batch_sampler
        has_fast_skip_api = hasattr(sampler, "_generate_batches") and hasattr(
            sampler, "_cached_generated_batches"
        )
        if skip_steps > 0 and has_fast_skip_api:
            all_batches = sampler._generate_batches(epoch)  # type: ignore[union-attr]  # noqa: SLF001
            remaining = all_batches[skip_steps:]
            # Temporarily override the sampler cache with the sliced list.
            sampler._cached_generated_batches = (  # type: ignore[union-attr]  # noqa: SLF001
                epoch,
                remaining,
            )
            root_logger.info(
                f"Fast-skipping {skip_steps} batches via sampler slice "
                f"(no vLLM calls for skipped batches). "
                f"epoch={epoch}, global_step={self.global_step}."
            )
        elif skip_steps > 0:
            root_logger.warning(
                "Sampler lacks fast-skip API; resume will replay "
                f"{skip_steps} batches from the start of the epoch."
            )
        return skip_steps

    @staticmethod
    def _format_worker_profile(step: int, worker: dict) -> str:
        identity = (
            f"step={step} rank={worker.get('rank', '?')} "
            f"local_rank={worker.get('local_rank', '?')} "
            f"worker={worker.get('worker_id', '?')} "
            f"pid={worker.get('worker_pid', '?')} "
            f"worker_task={worker.get('worker_task_seq', '?')} "
            f"sample={worker.get('sample_index', '?')} "
            f"seq_len={worker.get('seq_len', '?')} "
            f"source={worker.get('source', '?')}"
        )
        timing_keys = (
            "item_total_ms",
            "client_setup_ms",
            "dataset_payload_ms",
            "cache_lookup_ms",
            "cache_lock_check_ms",
            "cache_lock_wait_ms",
            "cache_exists_check_ms",
            "cache_stat_ms",
            "cache_file_load_ms",
            "cache_total_ms",
            "vllm_request_ms",
            "generated_read_ms",
            "generated_lock_check_ms",
            "generated_lock_wait_ms",
            "generated_exists_check_ms",
            "generated_stat_ms",
            "generated_file_load_ms",
            "generated_total_ms",
            "hidden_states_check_ms",
            "token_check_ms",
            "tensor_prepare_ms",
            "raw_data_total_ms",
            "sample_metadata_ms",
            "transform_ms",
            "preprocess_ms",
            "collate_ms",
            "collate_to_consume_ms",
            "cleanup_ms",
        )
        timings = " ".join(
            f"{key}={float(worker[key]):.2f}"
            for key in timing_keys
            if key in worker
        )
        details: list[str] = []
        for key in (
            "request_api",
            "client_reused",
            "vllm_endpoint",
            "cache_lock_present",
            "cache_file_bytes",
            "cache_path",
            "generated_lock_present",
            "generated_file_bytes",
            "cleanup_action",
            "sample_status",
            "handle",
            "error",
        ):
            if key in worker:
                details.append(f"{key}={worker[key]}")
        return f"[PIPELINE-WORKER] {identity} {timings} {' '.join(details)}"

    def _log_pipeline_profile(
        self,
        *,
        profile: dict[str, float],
        worker_profiles: list[dict],
        num_tokens: int,
    ) -> None:
        # Print details on the rank that produced them. The rank-0 filter is
        # explicitly overridden so rank 0 does not become a serial logging hub.
        log_extra = {"override_rank0_filter": True}
        rank_line = (
            "[PIPELINE-RANK] step=%d sync_ranks=%s rank=%d local_rank=%d "
            "tokens=%d loader_wait_ms=%.2f h2d_ms=%.2f "
            "pre_forward_sync_ms=%.2f forward_ms=%.2f backward_ms=%.2f "
            "optimizer_ms=%.2f step_ms=%.2f worker_samples=%d"
        ) % (
            self.global_step,
            self.config.profile_sync_ranks,
            self.rank,
            self.local_rank,
            num_tokens,
            profile["loader_wait_ms"],
            profile["h2d_ms"],
            profile["pre_fwd_sync_ms"],
            profile["fwd_ms"],
            profile["bwd_ms"],
            profile["opt_ms"],
            profile["step_ms"],
            len(worker_profiles),
        )
        workers = sorted(
            worker_profiles,
            key=lambda worker: (
                worker.get("worker_id", -1),
                worker.get("sample_index", -1),
            ),
        )
        local_lines = [rank_line]
        local_lines.extend(
            self._format_worker_profile(self.global_step, worker)
            for worker in workers
        )
        root_logger.warning("\n".join(local_lines), extra=log_extra)

        summary_freq = self.config.profile_summary_freq
        if summary_freq == 0 or self.global_step % summary_freq != 0:
            if self.is_distributed and self.config.profile_sync_ranks:
                # In sync-ranks mode, do not let different local log lengths
                # become next step's pre_forward_sync_ms.
                dist.barrier()
            return

        # Only fixed-size numeric tensors cross HCCL. Strings and variable-size
        # worker lists stay local to avoid all_gather_object compatibility and
        # serialization overhead.
        stage_keys = (
            "loader_wait_ms",
            "h2d_ms",
            "pre_fwd_sync_ms",
            "fwd_ms",
            "bwd_ms",
            "opt_ms",
            "step_ms",
        )
        worker_keys = (
            "item_total_ms",
            "vllm_request_ms",
            "cache_lookup_ms",
            "generated_read_ms",
            "generated_lock_wait_ms",
            "transform_ms",
            "collate_ms",
        )
        local_worker_maxima: list[dict | None] = []
        for key in worker_keys:
            candidates = [worker for worker in workers if key in worker]
            local_worker_maxima.append(
                max(candidates, key=lambda worker: float(worker[key]))
                if candidates
                else None
            )

        metric_values = [float(profile[key]) for key in stage_keys]
        metric_values.extend(
            float(worker[key]) if worker is not None else -1.0
            for key, worker in zip(worker_keys, local_worker_maxima, strict=True)
        )
        identity_values = [self.rank, self.local_rank, num_tokens, len(workers)]
        for worker in local_worker_maxima:
            identity_values.extend(
                [
                    int(worker.get("worker_id", -1)) if worker is not None else -1,
                    int(worker.get("sample_index", -1)) if worker is not None else -1,
                ]
            )

        profile_device = next(self.model.parameters()).device
        metric_tensor = torch.tensor(
            metric_values,
            dtype=torch.float32,
            device=profile_device,
        )
        identity_tensor = torch.tensor(
            identity_values,
            dtype=torch.int64,
            device=profile_device,
        )
        if self.is_distributed:
            world_size = dist.get_world_size()
            gathered_metrics = [
                torch.empty_like(metric_tensor) for _ in range(world_size)
            ]
            gathered_identities = [
                torch.empty_like(identity_tensor) for _ in range(world_size)
            ]
            dist.all_gather(gathered_metrics, metric_tensor)
            dist.all_gather(gathered_identities, identity_tensor)
        else:
            gathered_metrics = [metric_tensor]
            gathered_identities = [identity_tensor]

        if self.rank == 0:
            metric_rows = [tensor.cpu().tolist() for tensor in gathered_metrics]
            identity_rows = [tensor.cpu().tolist() for tensor in gathered_identities]

            def stage_extreme(key: str, *, minimum: bool = False) -> tuple[int, float]:
                column = stage_keys.index(key)
                chooser = min if minimum else max
                row = chooser(
                    range(len(metric_rows)),
                    key=lambda index: metric_rows[index][column],
                )
                return int(identity_rows[row][0]), float(metric_rows[row][column])

            def worker_extreme(key: str) -> str:
                worker_column = worker_keys.index(key)
                metric_column = len(stage_keys) + worker_column
                row = max(
                    range(len(metric_rows)),
                    key=lambda index: metric_rows[index][metric_column],
                )
                value = float(metric_rows[row][metric_column])
                if value < 0:
                    return "none"
                identity_column = 4 + worker_column * 2
                return (
                    f"rank{int(identity_rows[row][0])}/worker"
                    f"{int(identity_rows[row][identity_column])}/sample"
                    f"{int(identity_rows[row][identity_column + 1])}:{value:.2f}ms"
                )

            slow_loader = stage_extreme("loader_wait_ms")
            slow_h2d = stage_extreme("h2d_ms")
            slow_pre_fwd = stage_extreme("pre_fwd_sync_ms")
            fast_fwd = stage_extreme("fwd_ms", minimum=True)
            slow_fwd = stage_extreme("fwd_ms")
            slow_bwd = stage_extreme("bwd_ms")
            root_logger.warning(
                "[PIPELINE-SUMMARY] step=%d sync_ranks=%s "
                "loader_max=rank%d:%.2fms h2d_max=rank%d:%.2fms "
                "pre_forward_sync_max=rank%d:%.2fms "
                "forward_min=rank%d:%.2fms forward_max=rank%d:%.2fms "
                "backward_max=rank%d:%.2fms slow_item=%s max_vllm=%s "
                "max_cache_lookup=%s max_generated_read=%s max_file_lock=%s "
                "max_transform=%s max_collate=%s",
                self.global_step,
                self.config.profile_sync_ranks,
                *slow_loader,
                *slow_h2d,
                *slow_pre_fwd,
                *fast_fwd,
                *slow_fwd,
                *slow_bwd,
                worker_extreme("item_total_ms"),
                worker_extreme("vllm_request_ms"),
                worker_extreme("cache_lookup_ms"),
                worker_extreme("generated_read_ms"),
                worker_extreme("generated_lock_wait_ms"),
                worker_extreme("transform_ms"),
                worker_extreme("collate_ms"),
            )

        # The fixed-tensor gathers rendezvous after local detail output; this
        # final barrier only covers rank 0's single summary line so console I/O
        # cannot appear as another rank's next-step forward time.
        if self.is_distributed:
            dist.barrier()

    def train_epoch(self, epoch: int):
        self.model.train()
        if hasattr(self.train_loader.batch_sampler, "set_epoch"):
            self.train_loader.batch_sampler.set_epoch(epoch)  # type: ignore[union-attr]

        # Capture full-epoch step count before any resume fast-skip mutation.
        num_steps = len(self.train_loader)

        # Determine how many batches to skip for mid-epoch resume.
        skip_steps = self._prepare_resume_skip(epoch)

        train_loader = self.train_loader
        if self.rank == 0:
            train_loader = tqdm(train_loader, desc=f"Epoch {epoch}")  # type: ignore[assignment]

        step_interval = (
            max(1, round(num_steps * self.config.checkpoint_freq))
            if self.config.checkpoint_freq < 1
            else None
        )
        timer = _StepTimer()
        train_iterator = iter(train_loader)
        local_step_rel = 0
        while True:
            should_log_metrics = self.global_step % self.config.log_freq == 0
            timing_enabled = should_log_metrics or self.config.profile_pipeline
            timer.reset(timing_enabled)
            fetch_started = time.perf_counter() if timing_enabled else 0.0
            timer.mark_value("start", fetch_started)
            try:
                batch = next(train_iterator)
            except StopIteration:
                break
            data_ready = time.perf_counter() if timing_enabled else 0.0
            timer.mark_value("data_ready", data_ready)
            local_step_rel += 1

            # local_step is 1-based index into the *full* epoch (not the slice).
            local_step = local_step_rel + skip_steps
            worker_profiles = batch.pop(HS_PROFILE_KEY, [])
            for worker_profile in worker_profiles:
                collate_finished = worker_profile.pop(
                    "collate_finished_perf", None
                )
                if collate_finished is not None:
                    # perf_counter is system-wide on the training host. This
                    # interval is how long an already-collated batch takes to reach
                    # consumption: queue residence, IPC/deserialization, and the
                    # pin-memory path. A large value can mean useful prefetch lead;
                    # loader_wait_ms remains the measure of current-step blocking.
                    worker_profile["collate_to_consume_ms"] = max(
                        0.0,
                        (data_ready - float(collate_finished)) * 1000,
                    )
            gpu_batch = {
                k: v.to(self.local_rank, non_blocking=True)
                if isinstance(v, torch.Tensor)
                else v
                for k, v in batch.items()
            }

            with torch.autocast(
                self.device_type, dtype=self.config.hidden_states_dtype
            ):
                timer.mark("fetch")
                if (
                    self.config.profile_pipeline
                    and self.config.profile_sync_ranks
                    and self.is_distributed
                ):
                    # Make cross-rank data skew explicit. Without this diagnostic
                    # barrier, DDP buffer/parameter synchronization can charge time
                    # spent waiting for a data-straggler rank to "forward".
                    dist.barrier()
                    timer.mark("pre_fwd")
                _draft_tokens, loss, metrics = self.model(
                    **gpu_batch, **(self.config.train_call_kwargs or {})
                )

            timer.mark("fwd")
            self._optimizers_zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            timer.mark("bwd")
            self._optimizers_step()

            current_lrs = {
                type(opt).__name__: opt.param_groups[0]["lr"] for opt in self.optimizers
            }
            self._schedulers_step()
            timer.mark("opt")

            profile = None
            if timer.enabled:
                num_tokens = int((gpu_batch["document_ids"] != -1).sum().item())
                profile = timer.profile(num_tokens)
            else:
                num_tokens = 0

            if should_log_metrics:
                if self.is_distributed:
                    for v in metrics.values():
                        dist.reduce(v, dst=0, op=dist.ReduceOp.SUM)

                metrics = {k: v.item() for k, v in metrics.items()}
                world_size = dist.get_world_size() if self.is_distributed else 1
                metrics = normalize_counted_metrics(metrics, world_size)
                lr_info = (
                    current_lrs
                    if len(current_lrs) > 1
                    else next(iter(current_lrs.values()))
                )
                metric_logger.info(
                    {
                        "train": metrics,
                        "profile": profile,
                        "epoch": epoch,
                        "lr": lr_info,
                        "global_step": self.global_step,
                    },
                    extra={"step": self.global_step},
                )

            if self.config.profile_pipeline and profile is not None:
                self._log_pipeline_profile(
                    profile=profile,
                    worker_profiles=worker_profiles,
                    num_tokens=num_tokens,
                )
            self.global_step += 1

            if (
                step_interval is not None
                and not self.config.save_best
                and local_step % step_interval == 0
                and num_steps - local_step >= step_interval * MIN_STEP_PCT
                # Avoid saving back to back ay the end of each epoch
            ):
                self.maybe_save_checkpoint(epoch, local_step=local_step)

    @torch.no_grad()
    def val_epoch(self, epoch: int) -> dict[str, float] | None:
        if self.val_loader is None:
            return None
        self.model.eval()
        if hasattr(self.val_loader.batch_sampler, "set_epoch"):
            self.val_loader.batch_sampler.set_epoch(epoch)  # type: ignore[union-attr]
        val_loader = self.val_loader
        if self.rank == 0:
            val_loader = tqdm(val_loader, desc=f"Epoch {epoch}")  # type: ignore[assignment]

        val_metrics: dict[str, float] = {}
        num_batches = len(val_loader)
        for batch in val_loader:
            # Validation currently does not emit the detailed pipeline profile,
            # but ArrowDataset may still attach worker metadata when diagnostics
            # are enabled for the shared train/validation loader setup.
            batch.pop(HS_PROFILE_KEY, None)
            gpu_batch = {
                k: v.to(self.local_rank, non_blocking=True)
                if isinstance(v, torch.Tensor)
                else v
                for k, v in batch.items()
            }

            with torch.autocast(
                self.device_type, dtype=self.config.hidden_states_dtype
            ):
                _draft_tokens, _loss, metrics = self.model(
                    **gpu_batch, **(self.config.val_call_kwargs or {})
                )

            if self.is_distributed:
                for m in metrics.values():
                    dist.all_reduce(m, op=dist.ReduceOp.SUM)

            for k, v in metrics.items():
                val_metrics[k] = val_metrics.get(k, 0.0) + v.item()

        world_size = dist.get_world_size() if self.is_distributed else 1
        val_metrics = {k: v / num_batches for k, v in val_metrics.items()}
        val_metrics = normalize_counted_metrics(val_metrics, world_size)
        val_metrics = {f"{k}_epoch": v for k, v in val_metrics.items()}

        metric_logger.info(
            {"val": val_metrics, "epoch": epoch}, extra={"step": self.global_step}
        )

        return val_metrics

    def maybe_save_checkpoint(self, epoch: int | str, local_step: int = 0):
        if epoch != "interrupted" and (
            self.config.save_best
            or (
                self.config.checkpoint_freq >= 1
                and isinstance(epoch, int)
                and epoch != 0
                and (epoch + 1) % self.config.checkpoint_freq != 0
            )
        ):
            return

        root_logger.info(f"Saving checkpoint to {self.checkpointer.path / str(epoch)}")
        self.checkpointer.save_checkpoint(self.model, self.optimizers, epoch)
        if self.schedulers:
            self.checkpointer.save_scheduler_state_dict(self.schedulers, epoch)
        if isinstance(epoch, int):
            self._save_training_state(epoch, local_step)
            # Create a human-readable symlink for checkpoint readability.
            # e.g. epoch0_step16626 -> 0/ (mid) or epoch0_end -> 0/ (end)
            if not self.is_distributed or dist.get_rank() == 0:
                ckpt_dir = self.checkpointer.path
                suffix = f"step{local_step}" if local_step > 0 else "end"
                link_name = ckpt_dir / f"epoch{epoch}_{suffix}"
                target = Path(str(epoch))  # relative symlink
                # Remove any previous link for this epoch
                for old in ckpt_dir.glob(f"epoch{epoch}_*"):
                    if old.is_symlink():
                        old.unlink()
                link_name.symlink_to(target)
        root_logger.info(f"Checkpoint saved to {self.checkpointer.path / str(epoch)}")

    def maybe_update_best(self, epoch: int, val_metrics: dict | None):
        if val_metrics is None or "loss_epoch" not in val_metrics:
            return
        if val_metrics["loss_epoch"] >= self.best_val_loss:
            return

        if self.config.save_best:
            self.checkpointer.save_checkpoint(self.model, self.optimizers, epoch)
            if self.schedulers:
                self.checkpointer.save_scheduler_state_dict(self.schedulers, epoch)
        elif self.config.checkpoint_freq >= 1 and not (
            epoch == 0 or (epoch + 1) % int(self.config.checkpoint_freq) == 0
        ):
            return

        self.best_val_loss = val_metrics["loss_epoch"]
        self.checkpointer.save_val_metrics(epoch, val_metrics)
        self.checkpointer.update_best_symlink(epoch)
        root_logger.info(
            f"Updated checkpoint_best -> {epoch} (loss_epoch={self.best_val_loss:.6f})"
        )
        if self.config.save_best:
            self.checkpointer.cleanup_keep_only_best(best_epoch=epoch)

    @with_graceful_shutdown()
    def run_training(self):
        n_epochs = self.config.num_epochs
        for epoch in range(self.current_epoch, n_epochs):
            root_logger.info(f"Training epoch {epoch + 1}/{n_epochs} started")
            self.train_epoch(epoch)
            root_logger.info(f"Training epoch {epoch + 1}/{n_epochs} completed")

            if self.is_distributed:
                dist.barrier()

            self.maybe_save_checkpoint(epoch)

            if self.is_distributed:
                dist.barrier()

            val_metrics = None

            if self.val_loader is None:
                root_logger.warning("No val loader, skipping validation epoch")
            else:
                root_logger.info(f"Validation epoch {epoch + 1}/{n_epochs} started")
                val_metrics = self.val_epoch(epoch)
                root_logger.info(f"Validation epoch {epoch + 1}/{n_epochs} completed")

            if self.is_distributed:
                dist.barrier()

            self.maybe_update_best(epoch, val_metrics)

            if self.is_distributed:
                dist.barrier()
