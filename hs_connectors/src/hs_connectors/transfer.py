"""Abstraction for hidden-states transfer between vLLM and the trainer."""

from __future__ import annotations

import fcntl
import os
import shutil
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from safetensors.torch import load_file

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable


def wait_for_lock(lock_path: str, timeout: float = 10.0, poll_interval: float = 0.1):
    fd = os.open(lock_path, os.O_RDONLY)
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for lock: {lock_path}"
                    ) from None
                time.sleep(poll_interval)
    except BaseException:
        os.close(fd)
        raise
    os.close(fd)
    os.remove(lock_path)


class HiddenStatesTransfer(ABC):
    """Interface for reading hidden states produced by vLLM."""

    def setup(self) -> None:  # noqa: B027
        """Lazy initialization (safe to call from dataloader worker)."""

    @abstractmethod
    def get_cached(self, file_idx: int) -> dict[str, torch.Tensor] | None:
        """Return a previously cached sample, or ``None``."""

    @abstractmethod
    def get_generated(self, handle: str) -> dict[str, torch.Tensor] | None:
        """Retrieve a freshly generated sample by its vLLM-returned handle."""

    def cache(self, handle: str, file_idx: int) -> None:  # noqa: B027
        """Persist a generated sample to the cache location."""

    def delete(self, handle: str) -> None:  # noqa: B027
        """Clean up a generated sample (e.g. delete a temp file)."""

    def consume_profile(self) -> dict[str, Any]:
        """Return and clear timing details from the most recent transfer call.

        Backends that do not expose detailed timings may keep the default empty
        result.  The method is deliberately non-abstract so third-party transfer
        backends remain source compatible.
        """
        return {}

    def set_profile_enabled(self, enabled: bool) -> None:  # noqa: B027
        """Enable or disable backend timing collection."""


class HiddenStatesBackend(ABC):
    """Plugin interface for hidden-states transfer backends.

    Each backend registers itself via ``@HiddenStatesBackend.register(name)``
    and implements these four static hooks so that scripts (``train.py``,
    ``launch_vllm.py``) can discover and configure backends without hardcoding.
    """

    registry: ClassVar[dict[str, type[HiddenStatesBackend]]] = {}

    @classmethod
    def register(
        cls,
        name: str,
    ) -> Callable[[type[HiddenStatesBackend]], type[HiddenStatesBackend]]:
        def decorator(
            subclass: type[HiddenStatesBackend],
        ) -> type[HiddenStatesBackend]:
            if name in cls.registry:
                raise ValueError(f"Backend '{name}' is already registered.")
            cls.registry[name] = subclass
            return subclass

        return decorator

    @staticmethod
    @abstractmethod
    def add_train_args(parser: argparse.ArgumentParser) -> None:
        """Add backend-specific CLI arguments to ``train.py``."""
        ...

    @staticmethod
    @abstractmethod
    def add_launch_args(parser: argparse.ArgumentParser) -> None:
        """Add backend-specific CLI arguments to ``launch_vllm.py``."""
        ...

    @staticmethod
    @abstractmethod
    def from_train_args(
        args: argparse.Namespace,
        data_path: str,
    ) -> HiddenStatesTransfer:
        """Construct a :class:`HiddenStatesTransfer` from parsed train args."""
        ...

    @staticmethod
    @abstractmethod
    def build_kv_transfer_config(args: argparse.Namespace) -> dict[str, Any]:
        """Construct the ``kv_transfer_config`` dict for ``vllm serve``."""
        ...


# ---------------------------------------------------------------------------
# File-based backend (shared filesystem)
# ---------------------------------------------------------------------------


def _load_hs_file(
    file_path: Path,
    profile: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor] | None:
    # Preserve the original hot path exactly when profiling is disabled.  In
    # particular, avoid the extra stat() call because metadata operations can
    # be expensive on shared file systems.
    if profile is None:
        lock_path = str(file_path) + ".lock"
        if Path(lock_path).exists():
            wait_for_lock(lock_path)
        if file_path.exists():
            return load_file(file_path)
        return None

    started = time.perf_counter()
    profile.update(
        {
            "path": str(file_path),
            "lock_present": False,
            "lock_wait_ms": 0.0,
            "exists_check_ms": 0.0,
            "stat_ms": 0.0,
            "file_load_ms": 0.0,
            "file_bytes": 0,
        }
    )

    lock_path = str(file_path) + ".lock"
    lock_check_started = time.perf_counter()
    lock_present = Path(lock_path).exists()
    profile["lock_present"] = lock_present
    profile["lock_check_ms"] = (
        time.perf_counter() - lock_check_started
    ) * 1000
    if lock_present:
        lock_wait_started = time.perf_counter()
        try:
            wait_for_lock(lock_path)
        finally:
            profile["lock_wait_ms"] = (
                time.perf_counter() - lock_wait_started
            ) * 1000

    exists_started = time.perf_counter()
    file_exists = file_path.exists()
    profile["exists_check_ms"] = (time.perf_counter() - exists_started) * 1000
    profile["file_exists"] = file_exists

    if file_exists:
        stat_started = time.perf_counter()
        try:
            file_bytes = file_path.stat().st_size
        finally:
            profile["stat_ms"] = (time.perf_counter() - stat_started) * 1000
        profile["file_bytes"] = file_bytes

        load_started = time.perf_counter()
        try:
            return load_file(file_path)
        finally:
            profile["file_load_ms"] = (
                time.perf_counter() - load_started
            ) * 1000
            profile["total_ms"] = (time.perf_counter() - started) * 1000

    profile["total_ms"] = (time.perf_counter() - started) * 1000
    return None


class FileTransfer(HiddenStatesTransfer):
    """File-system based hidden-states transfer (shared filesystem)."""

    def __init__(self, hidden_states_path: Path):
        self.hidden_states_path = hidden_states_path
        self.profile_enabled = False
        self._last_profile: dict[str, Any] = {}

    def _load(self, path: Path) -> dict[str, torch.Tensor] | None:
        if not self.profile_enabled:
            return _load_hs_file(path)
        profile: dict[str, Any] = {}
        try:
            return _load_hs_file(path, profile)
        finally:
            self._last_profile = profile

    def get_cached(self, file_idx: int) -> dict[str, torch.Tensor] | None:
        path = self.hidden_states_path / f"hs_{file_idx}.safetensors"
        return self._load(path)

    def get_generated(self, handle: str) -> dict[str, torch.Tensor] | None:
        return self._load(Path(handle))

    def consume_profile(self) -> dict[str, Any]:
        profile = self._last_profile
        self._last_profile = {}
        return profile

    def set_profile_enabled(self, enabled: bool) -> None:
        if self.profile_enabled == enabled:
            return
        self.profile_enabled = enabled
        if not enabled:
            self._last_profile = {}

    def cache(self, handle: str, file_idx: int) -> None:
        self.hidden_states_path.mkdir(parents=True, exist_ok=True)
        target = self.hidden_states_path / f"hs_{file_idx}.safetensors"
        shutil.move(handle, target)

    def delete(self, handle: str) -> None:
        Path(handle).unlink()


@HiddenStatesBackend.register("file")
class FileBackend(HiddenStatesBackend):
    """Shared-filesystem backend using safetensors files."""

    @staticmethod
    def add_train_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--hidden-states-path",
            type=str,
            default=None,
            help=(
                "The path where cached hidden states files are stored. (Default: "
                "args.data_path / 'hidden_states')"
            ),
        )

    @staticmethod
    def add_launch_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--hidden-states-path",
            type=str,
            default="/tmp/hidden_states",  # noqa: S108
            help="The directory to save hidden states to. Default '/tmp/hidden_states'",
        )

    @staticmethod
    def from_train_args(
        args: argparse.Namespace,
        data_path: str,
    ) -> FileTransfer:
        hs_path = (
            Path(args.hidden_states_path)
            if args.hidden_states_path
            else Path(data_path) / "hidden_states"
        )
        return FileTransfer(hs_path)

    @staticmethod
    def build_kv_transfer_config(args: argparse.Namespace) -> dict[str, Any]:
        return {
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "shared_storage_path": args.hidden_states_path,
            },
        }
