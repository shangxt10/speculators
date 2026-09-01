import json
import math
import os
import random
import time
import warnings
from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import Any, Literal, cast

import openai
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset, get_worker_info

from hs_connectors import FileTransfer, HiddenStatesTransfer
from speculators.data_generation.offline import check_hidden_states
from speculators.data_generation.vllm_client import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT,
    ClientItem,
    generate_hidden_states,
)
from speculators.train.noise_transforms import TransformTensors

BatchType = dict[str, Any]
HS_PROFILE_KEY = "__hs_profile__"
HS_PROFILE_ONLY_KEY = "__hs_profile_only__"


def list_files(path):
    datapath = []
    for root, _directories, files in os.walk(path):
        for file in files:
            if not file.endswith("pt"):
                continue
            file_path = Path(root) / file
            datapath.append(file_path)

    return datapath


def split_files(datapath: str, ratio: float = 0.9, seed: int = 0):
    """Given a datapath, split the files into a training and validation set
    ratio is the proportion of files to put in the training set
    1 - ratio is the proportion of files to put in the validation set
    """
    random.seed(seed)
    file_list = list_files(datapath)
    random.shuffle(file_list)
    num_files = len(file_list)
    num_train_files = int(num_files * ratio)
    train_files = file_list[:num_train_files]
    val_files = file_list[num_train_files:]
    return train_files, val_files


# Data standardization functions
StandardizeFnSig = Callable[[dict[str, Any]], dict[str, Any]]


def create_empty_sample(
    hidden_size: int, num_target_layers: int = 3, dtype: torch.dtype = torch.bfloat16
):
    # data structure: {
    #     "hidden_states": [seq_len, num_target_layers * hidden_size],
    #     "input_ids": [seq_len],
    #     "verifier_last_hidden_states": [seq_len, hidden_size],
    #     "loss_mask": [seq_len],
    #     "lengths": [1],
    #     "position_ids": [seq_len],
    # }
    # Default dtype is bfloat16 to match the hidden_states dtype used downstream.
    # When this fallback is used (e.g. vLLM hidden-state extraction times out and
    # we substitute an empty sample), the implicit float32 placeholders crashed
    # bf16 EAGLE-3 layers (fc, verifier_lm_head) with a dtype mismatch.

    return {
        "hidden_states": torch.empty(0, num_target_layers * hidden_size, dtype=dtype),
        "input_ids": torch.empty(0, dtype=torch.long),
        "verifier_last_hidden_states": torch.empty(0, hidden_size, dtype=dtype),
        "loss_mask": torch.empty(0, dtype=torch.bool),
        "lengths": torch.tensor([0], dtype=torch.long),
        "position_ids": torch.arange(0, dtype=torch.long),
    }


def standardize_data_v1(data: dict[str, Any]) -> dict[str, Any]:
    # v1 data format:
    # {
    #  "input_ids": [seq_len],
    #  "loss_mask": [seq_len],
    #  "hidden_states": [
    #    [seq_len, hidden_size],
    #    [seq_len, hidden_size],
    #    [seq_len, hidden_size],
    #    ...
    #  ],
    # }

    return {
        "hidden_states": torch.cat(data["hidden_states"][:-1], dim=-1),
        "input_ids": data["input_ids"],
        "verifier_last_hidden_states": data["hidden_states"][-1],
        "loss_mask": data["loss_mask"],
    }


def _has_multimodal_content(messages: list[dict]) -> bool:
    """True when any turn carries non-text content (images, video, audio).

    Text-only turns store ``content`` as a plain string.  Multimodal turns
    (produced by ``_adapt_conv_for_vllm``) store it as a list of typed parts,
    e.g. ``[{"type": "text", ...}, {"type": "image_url", ...}]``.
    """
    return any(isinstance(m.get("content"), list) for m in messages)


def build_client_item(dataset_item: dict) -> ClientItem:
    """Build a request payload for vLLM hidden-state extraction.

    When ``messages`` is included, ``generate_hidden_states`` uses the Chat
    Completions API and vLLM **re-tokenizes from the raw messages**, ignoring
    ``input_ids``.  This is required for multimodal inputs (the Completions
    API cannot carry image/video/audio references), but harmful for text-only
    data: preprocessing truncates ``input_ids`` to ``seq_length``, yet the
    ``messages`` column stores the original un-truncated conversation.
    Re-tokenizing those messages produces a longer sequence that can exceed
    ``max_model_len``.

    We therefore only forward ``messages`` when the conversation actually
    contains multimodal content.  Text-only conversations always go through
    the Completions API with the pre-truncated ``input_ids``.

    This matters for models like Qwen3.5-0.8B whose ``AutoProcessor`` returns
    a ``ProcessorMixin`` (``Qwen3VLProcessor``), causing preprocessing to
    populate the ``messages`` column even for purely text-only datasets.
    Text-only EAGLE-3 models (e.g. Llama) use a plain tokenizer, so
    ``messages`` is never created and this guard is a no-op.
    """
    out_dict: dict = {"input_ids": dataset_item["input_ids"].tolist()}

    if "messages" in dataset_item and _has_multimodal_content(dataset_item["messages"]):
        out_dict["messages"] = dataset_item["messages"]

    return cast("ClientItem", out_dict)


class BaseDataset(Dataset):
    def __init__(
        self,
        max_len: int,
        transform: TransformTensors | None = None,
        hidden_states_dtype=torch.bfloat16,
    ):
        self.max_len = max_len
        self.transform = transform
        self.hidden_states_dtype = hidden_states_dtype
        self.approx_lengths = self._compute_approx_lengths()

    def _compute_approx_lengths(self):
        raise NotImplementedError

    def _get_raw_data(self, index):
        raise NotImplementedError

    def __getitem__(self, index) -> BatchType | None:
        profile = getattr(self, "_active_profile", None)
        raw_started = time.perf_counter() if profile is not None else 0.0
        data = self._get_raw_data(index)
        if profile is not None:
            profile["raw_data_total_ms"] = (
                time.perf_counter() - raw_started
            ) * 1000

        if data is None:
            return data

        # data structure: {
        #  "hidden_states": [seq_len, 3 * hidden_size],
        #  "input_ids": [seq_len],
        #  "verifier_last_hidden_states": [seq_len, hidden_size],
        #  "loss_mask": [seq_len],
        # }

        metadata_started = time.perf_counter() if profile is not None else 0.0
        # Add lengths tensor
        seq_len = data["input_ids"].shape[0]
        data["lengths"] = torch.tensor([seq_len], dtype=torch.long)
        # shape: [1]

        data["position_ids"] = torch.arange(seq_len, dtype=torch.long)
        # shape: [seq_len]

        # data structure: {
        #     "hidden_states": [seq_len, 3 * hidden_size],
        #     "input_ids": [seq_len],
        #     "verifier_last_hidden_states": [seq_len, hidden_size],
        #     "loss_mask": [seq_len],
        #     "lengths": [1],
        #     "position_ids": [seq_len],
        # }

        if profile is not None:
            profile["sample_metadata_ms"] = (
                time.perf_counter() - metadata_started
            ) * 1000

        # Apply transform. For mmap-backed safetensors this may be where file
        # pages are first touched, so report it separately from load_file().
        if self.transform:
            transform_started = time.perf_counter() if profile is not None else 0.0
            try:
                data = self.transform(data)
            finally:
                if profile is not None:
                    profile["transform_ms"] = (
                        time.perf_counter() - transform_started
                    ) * 1000
        elif profile is not None:
            profile["transform_ms"] = 0.0

        return data


class ArrowDataset(BaseDataset):
    def __init__(
        self,
        max_len: int,
        datapath: str | PathLike,
        transfer: HiddenStatesTransfer | None = None,
        vllm_endpoint: str = "http://localhost:8000/v1",
        on_missing: Literal["generate", "skip", "warn", "raise"] = "generate",
        on_generate: Literal["cache", "delete"] = "delete",
        split_ratio: float = 1.0,
        transform: TransformTensors | None = None,
        hidden_states_dtype=torch.bfloat16,
        model: str | None = None,
        request_timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        profile_pipeline: bool = False,
    ):
        self.data = load_from_disk(datapath)
        self.start_file_idx = 0
        if split_ratio == 1.0:
            pass
        elif 1.0 > split_ratio > 0:
            self.start_file_idx = 0
            split_idx = int(len(self.data) * split_ratio)
            self.data = self.data.select(range(split_idx))
        elif -1.0 < split_ratio < 0:
            split_idx = int(len(self.data) * (1.0 + split_ratio))
            self.start_file_idx = split_idx
            self.data = self.data.select(range(split_idx, len(self.data)))
        else:
            raise ValueError("split_ratio must be in range (-1.0, 1.0] excluding 0.0.")

        self.transfer = transfer or FileTransfer(Path(datapath) / "hidden_states")
        self.vllm_endpoint = vllm_endpoint
        self.on_missing = on_missing
        self.on_generate = on_generate
        self.client: openai.OpenAI | None = None
        self.model = model
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.profile_pipeline = profile_pipeline
        self._active_profile: dict[str, Any] | None = None
        self._profile_item_count = 0

        # Delay super init so that `_compute_approx_lengths` has required data
        super().__init__(max_len, transform, hidden_states_dtype)

    def _map_to_file_idx(self, index: int):
        return index + self.start_file_idx

    def __getitem__(self, index) -> BatchType | None:
        if not self.profile_pipeline:
            self.transfer.set_profile_enabled(False)
            return super().__getitem__(index)

        worker = get_worker_info()
        self._profile_item_count += 1
        profile: dict[str, Any] = {
            "rank": int(os.environ.get("RANK", "0")),
            "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
            "worker_id": worker.id if worker is not None else -1,
            "worker_pid": os.getpid(),
            "worker_task_seq": self._profile_item_count,
            "sample_index": int(index),
            "file_index": int(self._map_to_file_idx(index)),
            "vllm_endpoint": self.vllm_endpoint,
        }
        self._active_profile = profile
        self.transfer.set_profile_enabled(True)
        started = time.perf_counter()
        try:
            result = super().__getitem__(index)
            profile["item_total_ms"] = (time.perf_counter() - started) * 1000
            if result is None:
                # Preserve timing for failed/skipped samples without letting the
                # profile-only record enter preprocessing or model tensors.
                profile["sample_status"] = "dropped"
                return {
                    HS_PROFILE_KEY: profile,
                    HS_PROFILE_ONLY_KEY: True,
                }
            profile["sample_status"] = "ok"
            result[HS_PROFILE_KEY] = profile
            return result
        except Exception as error:
            profile["item_total_ms"] = (time.perf_counter() - started) * 1000
            profile["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            self._active_profile = None

    def _record_transfer_profile(self, prefix: str) -> None:
        if self._active_profile is None:
            return
        details = self.transfer.consume_profile()
        self._active_profile.update(
            {f"{prefix}_{key}": value for key, value in details.items()}
        )

    def _setup_client(self):
        self.client = openai.OpenAI(
            base_url=self.vllm_endpoint, api_key="EMPTY", max_retries=0
        )
        list_models = self.client.models.list()
        model_id = list_models.data[0].id
        if self.model and self.model != model_id:
            raise ValueError(
                f"An explicit model name was passed ({self.model}) which doesn't match"
                f" found model_id {model_id}."
                "Please make sure --endpoint is set to the correct vllm instance."
            )
        self.model = model_id
        self.transfer.setup()

    def __len__(self):
        return len(self.data)

    def _compute_approx_lengths(self) -> list[int]:
        """Get lengths of the dataset samples."""
        return list(self.data.with_format(None)["seq_len"])

    def _maybe_generate_hs(self, index: int) -> dict[str, torch.Tensor] | None:
        profile = self._active_profile
        client_reused = self.client is not None
        if profile is not None:
            profile["client_reused"] = client_reused
        if not self.client:
            setup_started = time.perf_counter() if profile is not None else 0.0
            try:
                self._setup_client()
            finally:
                if profile is not None:
                    profile["client_setup_ms"] = (
                        time.perf_counter() - setup_started
                    ) * 1000

        dataset_started = time.perf_counter() if profile is not None else 0.0
        dataset_item = self.data[index]
        client_item = build_client_item(dataset_item)
        if profile is not None:
            profile["dataset_payload_ms"] = (
                time.perf_counter() - dataset_started
            ) * 1000
            profile["seq_len"] = len(client_item["input_ids"])
            profile["request_api"] = (
                "chat" if client_item.get("messages") is not None else "completions"
            )

        try:
            request_started = time.perf_counter() if profile is not None else 0.0
            try:
                handle = generate_hidden_states(
                    self.client,  # type:ignore[arg-type]
                    self.model,  # type:ignore[arg-type]
                    client_item,
                    timeout=self.request_timeout,
                    max_retries=self.max_retries,
                )
            finally:
                if profile is not None:
                    profile["vllm_request_ms"] = (
                        time.perf_counter() - request_started
                    ) * 1000

            if profile is not None:
                profile["handle"] = str(handle)

            generated_read_started = (
                time.perf_counter() if profile is not None else 0.0
            )
            try:
                loaded_hs = self.transfer.get_generated(handle)
            finally:
                if profile is not None:
                    profile["generated_read_ms"] = (
                        time.perf_counter() - generated_read_started
                    ) * 1000
                self._record_transfer_profile("generated")
            if loaded_hs is None:
                raise ValueError(f"Failed to load hidden states for handle {handle}")

            check_started = time.perf_counter() if profile is not None else 0.0
            try:
                check_hidden_states(loaded_hs, dataset_item["input_ids"].tolist())
            finally:
                if profile is not None:
                    profile["hidden_states_check_ms"] = (
                        time.perf_counter() - check_started
                    ) * 1000

            file_idx = self._map_to_file_idx(index)
            cleanup_started = time.perf_counter() if profile is not None else 0.0
            try:
                match self.on_generate:
                    case "cache":
                        self.transfer.cache(handle, file_idx)
                    case "delete":
                        self.transfer.delete(handle)
            finally:
                if profile is not None:
                    profile["cleanup_ms"] = (
                        time.perf_counter() - cleanup_started
                    ) * 1000
                    profile["cleanup_action"] = self.on_generate
        except Exception as e:
            if profile is not None:
                profile["error"] = f"{type(e).__name__}: {e}"
            if isinstance(e, ValueError) and "NaN" in str(e):
                raise
            warnings.warn(
                f"Failed to load/cache hidden states for sample {index}: {e}. "
                f"profile={profile if profile is not None else 'disabled'}",
                stacklevel=1,
            )
            return None

        return loaded_hs

    def _get_raw_data(self, index):
        profile = self._active_profile
        file_idx = self._map_to_file_idx(index)
        cache_started = time.perf_counter() if profile is not None else 0.0
        try:
            loaded_hs = self.transfer.get_cached(file_idx)
        finally:
            if profile is not None:
                profile["cache_lookup_ms"] = (
                    time.perf_counter() - cache_started
                ) * 1000
            self._record_transfer_profile("cache")

        if loaded_hs is None:
            if profile is not None:
                profile["source"] = "generated"
            match self.on_missing:
                case "generate":
                    loaded_hs = self._maybe_generate_hs(index)
                case "skip":
                    return None
                case "warn":
                    warnings.warn(
                        f"Failed to load hidden states for sample {index}. Skipping...",
                        stacklevel=1,
                    )
                    return None
                case "raise":
                    raise RuntimeError(
                        f"Failed to load hidden states for sample {index}."
                    )
        elif profile is not None:
            profile["source"] = "cache"

        if loaded_hs is None:
            return loaded_hs

        if profile is not None:
            profile["seq_len"] = int(loaded_hs["token_ids"].numel())

        # loaded_hs structure: {
        #   "hidden_states": [seq_len, num_layers, hidden_size]
        #   "token_ids": [seq_len]
        # }

        token_check_started = time.perf_counter() if profile is not None else 0.0
        token_ids_match = torch.equal(
            loaded_hs["token_ids"], self.data[index]["input_ids"]
        )
        if profile is not None:
            profile["token_check_ms"] = (
                time.perf_counter() - token_check_started
            ) * 1000
        if not token_ids_match:
            warnings.warn(
                f"Loaded token ids {loaded_hs['token_ids']} for index {index} don't"
                f"match input ids {self.data[index]['input_ids']}",
                stacklevel=1,
            )
            return None

        tensor_prepare_started = time.perf_counter() if profile is not None else 0.0
        try:
            return {
                "hidden_states": loaded_hs["hidden_states"][:, :-1].flatten(
                    1
                ),  # [seq_len, 3 * hidden_size]
                "input_ids": loaded_hs["token_ids"],  # [seq_len]
                "verifier_last_hidden_states": loaded_hs["hidden_states"][
                    :, -1
                ],  # [seq_len, hidden_size]
                "loss_mask": self.data[index]["loss_mask"],  # [seq_len]
            }
        finally:
            if profile is not None:
                profile["tensor_prepare_ms"] = (
                    time.perf_counter() - tensor_prepare_started
                ) * 1000


class SampleFileDataset(BaseDataset):
    def __init__(
        self,
        max_len: int,
        datapath: str | None = None,
        file_list: list[str] | None = None,
        transform: TransformTensors | None = None,
        hidden_states_dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize the SampleFileDataset.
        Args:
            max_len: The maximum length of the sequence.
            datapath: The path to the data directory. All `.pt` files in this directory
            or its subdirectories will be loaded and used as training data. MUTUALLY
            EXCLUSIVE with `file_list`.
            file_list: The list of explict file paths to load data from. These files
            must be in the format produced by the Speculators generation scripts.
            MUTUALLY EXCLUSIVE with `datapath`.
            transform: The transform to apply to the data.
            hidden_states_dtype: The dtype of the hidden states.
            standardize_fn: The function to standardize the data.

            Note: datapath or file_list must be provided, but not both.

        """

        if datapath is not None and file_list is not None:
            raise ValueError(
                "Either `datapath` or `file_list` must be provided, but "
                "not both. Use `datapath` to auto-discover files, or "
                "`file_list` to use a list of explicit file paths."
            )

        if datapath is not None:
            file_list = list_files(datapath)

        if file_list is None:
            raise ValueError(
                "Either `datapath` or `file_list` must be provided, but "
                "not both. Use `datapath` to auto-discover files, or "
                "`file_list` to use a list of explicit file paths."
            )

        self.data: list[str] = file_list

        # Delay super init so that `_compute_approx_lengths` has required data
        super().__init__(max_len, transform, hidden_states_dtype)

    def __len__(self):
        return len(self.data)

    def _compute_approx_lengths(self) -> list[int]:
        """Get lengths of the dataset samples.

        First tries to load exact lengths from sample_lengths.json if available.
        Falls back to approximation based on file sizes.
        """
        # Look for the sample_lengths.json file
        sample_lengths_path = Path(self.data[0]).parent / "sample_lengths.json"
        if sample_lengths_path.exists():
            try:
                with sample_lengths_path.open() as f:
                    sample_lengths = json.load(f)
                # Extract file index from filename (e.g., data_42.pt -> 42)
                lengths = []
                for fname in self.data:
                    file_stem = Path(fname).stem
                    file_idx = file_stem.split("_")[-1]
                    lengths.append(sample_lengths[file_idx])
                return lengths
            except (KeyError, ValueError):
                pass

        # Fallback: approximate lengths from file sizes
        item_0 = self.__getitem__(0)
        if item_0 is None:
            raise ValueError(
                "Failed to load first element of datasets for length approximation"
            )
        lengths_0 = item_0["lengths"]
        # this is a single sample so there is only one length
        lengths_0 = lengths_0[0].item()
        size_0 = Path(self.data[0]).stat().st_size

        return [
            math.ceil(Path(fname).stat().st_size / size_0 * lengths_0)
            for fname in self.data
        ]

    def _get_raw_data(self, index):
        return standardize_data_v1(
            torch.load(
                self.data[index], mmap=True, weights_only=True, map_location="cpu"
            )
        )


def create_collate_fn(
    max_len: int,
    hidden_size: int,
    num_target_layers: int = 3,
    dtype: torch.dtype = torch.bfloat16,
    preprocess: Callable[[BatchType], BatchType] | None = None,
):
    def collate_fn(batch: list[BatchType | None]) -> BatchType:
        # Apply per-sample preprocessing and filter failed samples
        profiles: list[dict[str, Any]] = []
        data_batch: list[BatchType] = []
        for sample in batch:
            if sample is None:
                continue
            profile = sample.pop(HS_PROFILE_KEY, None)
            if profile is not None:
                profiles.append(profile)
            if sample.pop(HS_PROFILE_ONLY_KEY, False):
                continue
            preprocess_started = time.perf_counter() if profile is not None else 0.0
            processed = preprocess(sample) if preprocess else sample
            if profile is not None:
                profile["preprocess_ms"] = (
                    time.perf_counter() - preprocess_started
                ) * 1000
            data_batch.append(processed)
        batch = data_batch
        collate_started = time.perf_counter() if profiles else 0.0

        if not batch:
            # Create empty sample which then gets padded to full
            # batch size if no valid samples are found.
            # Match the configured `dtype` so the placeholder doesn't crash
            # downstream layers loaded at a different precision (e.g. bf16
            # weights vs fp32 default placeholders).
            empty = create_empty_sample(hidden_size, num_target_layers, dtype=dtype)
            if preprocess:
                empty = preprocess(empty)
            batch = [empty]

        collated_data = {}
        for key in batch[0]:  # type: ignore[union-attr]
            if key == "lengths":
                collated_data[key] = torch.cat([b[key] for b in batch], dim=0)  # type: ignore[index]
                continue
            # one copy per sample: preallocated buffer, hidden states cast during write
            first = batch[0][key]  # type: ignore[index]
            buffer_dtype = dtype if "hidden_states" in key else first.dtype
            out = torch.zeros(
                (max_len, *first.shape[1:]), dtype=buffer_dtype, device=first.device
            )
            offset = 0
            for b in batch:
                tensor = b[key]  # type: ignore[index]
                num_rows = min(tensor.shape[0], max_len - offset)
                out[offset : offset + num_rows] = tensor[:num_rows]
                offset += num_rows
                if offset == max_len:
                    break
            collated_data[key] = out.unsqueeze(0)
            # shape: [1, max_len, ...]

        # Include lengths until while they fit in max_len
        # The last included length is (if necessary) truncated
        # Any additional lengths are discarded
        lengths = collated_data.pop("lengths")
        new_lengths = []
        cum_length = 0
        for length in lengths:
            if length + cum_length >= max_len:
                new_lengths.append(max_len - cum_length)
                break
            new_lengths.append(length)
            cum_length += length
        lengths = torch.tensor(new_lengths, dtype=torch.long)

        # Create document_ids: maps each position to its document index, -1 for padding
        document_ids = torch.repeat_interleave(
            torch.arange(lengths.shape[0], dtype=torch.long), lengths
        )
        document_ids = torch.cat(
            [
                document_ids,
                -1 * torch.ones(max_len - document_ids.shape[0], dtype=torch.long),
            ]
        ).unsqueeze(0)
        # shape: [1, max_len]
        collated_data["document_ids"] = document_ids
        if profiles:
            collate_finished = time.perf_counter()
            collate_ms = (collate_finished - collate_started) * 1000
            for profile in profiles:
                profile["collate_ms"] = collate_ms
                profile["collate_finished_perf"] = collate_finished
            collated_data[HS_PROFILE_KEY] = profiles

        return collated_data

    return collate_fn
