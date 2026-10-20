"""Leakage-free data utilities for the ASTAF-R revision experiments.

The submitted code normalized the complete time series before splitting.  This
module instead computes normalization statistics from the chronological
training period only, retains a target-validity mask for values encoded as
zero, and exposes sliding windows without materializing the full X/y tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pickle
from typing import Dict, Tuple
from zoneinfo import ZoneInfo

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    raw: np.ndarray
    normalized: np.ndarray
    valid: np.ndarray
    adjacency: np.ndarray
    adjacency_source: str
    mean: float
    std: float
    split_indices: Dict[str, np.ndarray]
    timestamps: np.ndarray
    split_time_bounds: Dict[str, Tuple[int, int]]
    calendar_positions: np.ndarray
    seq_length: int
    horizon: int


class TrafficWindowDataset(Dataset):
    def __init__(self, bundle: DatasetBundle, starts: np.ndarray):
        self.bundle = bundle
        self.starts = np.asarray(starts, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.starts.size)

    def __getitem__(self, index: int):
        start = int(self.starts[index])
        middle = start + self.bundle.seq_length
        end = middle + self.bundle.horizon
        x = self.bundle.normalized[start:middle, :, None]
        y = self.bundle.normalized[middle:end, :, None]
        mask = self.bundle.valid[middle:end, :, None]
        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(mask),
            # Local-calendar positions preserve the observed time of day. This
            # matters for PeMS-BAY because the series crosses the March 2017
            # daylight-saving transition in California.
            torch.tensor(self.bundle.calendar_positions[start], dtype=torch.int64),
        )


def _load_raw(
    data_dir: Path, dataset_name: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if dataset_name == "METR-LA":
        with h5py.File(data_dir / "metr-la.h5", "r") as handle:
            raw = handle["df/block0_values"][:].astype(np.float32)
            timestamp_values = handle["df/axis1"][:].astype("datetime64[ns]")
        day_offsets = (
            timestamp_values.astype("datetime64[D]")
            - timestamp_values[0].astype("datetime64[D]")
        ).astype(np.int64)
        minute_of_day = (
            timestamp_values.astype("datetime64[m]")
            - timestamp_values.astype("datetime64[D]")
        ).astype(np.int64)
        calendar_positions = day_offsets * 288 + minute_of_day // 5
        timestamps = np.asarray(
            [np.datetime_as_string(value, unit="m") for value in timestamp_values],
            dtype="U32",
        )
        adjacency_path = data_dir / "adj_mx_METR-LA.pkl"
        if adjacency_path.exists():
            with adjacency_path.open("rb") as stream:
                adjacency_object = pickle.load(stream, encoding="latin1")
            adjacency = (
                adjacency_object[-1]
                if isinstance(adjacency_object, (list, tuple))
                else adjacency_object
            )
            adjacency = np.asarray(adjacency, dtype=np.float32)
        else:
            processed = np.load(data_dir / "processed" / "METR-LA_processed.npz")
            adjacency = processed["adj_matrix"].astype(np.float32)
    elif dataset_name == "PeMS-BAY":
        values = np.load(data_dir / "pems_node_values.npy", mmap_mode="r")
        raw = np.asarray(values[:, :, 0], dtype=np.float32)
        adjacency = np.load(data_dir / "pems_adj_mat.npy").astype(np.float32)
        # The distributed array begins at 00:55 Pacific Standard Time on
        # 1 January 2017. Construct the underlying regular UTC sequence and
        # convert each observation to California local time. This reproduces
        # the one-hour clock advance on 12 March rather than silently shifting
        # all later calendar features.
        pacific = ZoneInfo("America/Los_Angeles")
        start_utc = datetime(2017, 1, 1, 8, 55, tzinfo=timezone.utc)
        local_times = [
            (start_utc + timedelta(minutes=5 * index)).astimezone(pacific)
            for index in range(raw.shape[0])
        ]
        first_date = local_times[0].date()
        calendar_positions = np.asarray(
            [
                (stamp.date() - first_date).days * 288
                + (stamp.hour * 60 + stamp.minute) // 5
                for stamp in local_times
            ],
            dtype=np.int64,
        )
        timestamps = np.asarray(
            [stamp.isoformat(timespec="minutes") for stamp in local_times], dtype="U32"
        )
        expected_tod = (calendar_positions % 288) / 288.0
        observed_tod = np.asarray(values[:, 0, 1], dtype=np.float64)
        if not np.allclose(observed_tod, expected_tod, atol=1e-7):
            raise ValueError("PeMS-BAY time-of-day channel does not match timestamps")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return raw, adjacency, timestamps, calendar_positions


def prepare_dataset(
    data_dir: str | Path,
    dataset_name: str,
    seq_length: int = 12,
    horizon: int = 12,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> DatasetBundle:
    """Load a benchmark and prepare chronological train/val/test windows.

    Values equal to zero are treated as unavailable observations, matching the
    masking convention used by common traffic-forecasting evaluation code.
    They are mean-imputed in model inputs (normalized value 0) and excluded
    from training loss and metrics when they occur in targets.
    """

    data_dir = Path(data_dir)
    raw, adjacency, timestamps, calendar_positions = _load_raw(data_dir, dataset_name)
    valid = np.isfinite(raw) & (raw != 0)

    train_time_end = int(raw.shape[0] * train_ratio)
    train_values = raw[:train_time_end][valid[:train_time_end]]
    mean = float(train_values.mean())
    std = float(train_values.std())
    if std <= 0:
        raise ValueError(f"Non-positive training standard deviation for {dataset_name}")

    normalized = ((raw - mean) / std).astype(np.float32)
    normalized[~valid] = 0.0

    if dataset_name == "METR-LA" and (data_dir / "adj_mx_METR-LA.pkl").exists():
        adjacency_source = "standard DCRNN-compatible METR-LA distance adjacency"
    elif dataset_name == "METR-LA":
        adjacency_source = "legacy processed-package correlation adjacency"
    else:
        adjacency_source = "provided PeMS-BAY adjacency matrix"

    # Partition the raw timeline first, then construct windows independently
    # inside each segment. Consequently, no raw observation can appear in more
    # than one of train, validation, and test, even as an input in one split
    # and a target in another.
    train_end = int(raw.shape[0] * train_ratio)
    val_end = int(raw.shape[0] * (train_ratio + val_ratio))
    split_time_bounds = {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, raw.shape[0]),
    }
    window_span = seq_length + horizon
    split_indices: Dict[str, np.ndarray] = {}
    for split, (segment_start, segment_end) in split_time_bounds.items():
        stop = segment_end - window_span + 1
        if stop <= segment_start:
            raise ValueError(f"{split} segment is too short for a {window_span}-step window")
        split_indices[split] = np.arange(segment_start, stop, dtype=np.int64)

    return DatasetBundle(
        name=dataset_name,
        raw=raw,
        normalized=normalized,
        valid=valid,
        adjacency=adjacency,
        adjacency_source=adjacency_source,
        mean=mean,
        std=std,
        split_indices=split_indices,
        timestamps=timestamps,
        split_time_bounds=split_time_bounds,
        calendar_positions=calendar_positions,
        seq_length=seq_length,
        horizon=horizon,
    )


def make_loaders(
    bundle: DatasetBundle,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    loaders: Dict[str, DataLoader] = {}
    for split, starts in bundle.split_indices.items():
        loaders[split] = DataLoader(
            TrafficWindowDataset(bundle, starts),
            batch_size=batch_size,
            shuffle=(split == "train"),
            generator=generator if split == "train" else None,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
    return loaders


def dataset_summary(bundle: DatasetBundle) -> dict:
    def timestamp(index: int) -> str:
        return str(bundle.timestamps[index])

    boundaries = {}
    for split, (segment_start, segment_end) in bundle.split_time_bounds.items():
        starts = bundle.split_indices[split]
        first_start = int(starts[0])
        last_start = int(starts[-1])
        boundaries[split] = {
            "raw_segment_start_index": segment_start,
            "raw_segment_end_index_exclusive": segment_end,
            "raw_segment_start_timestamp": timestamp(segment_start),
            "raw_segment_end_timestamp": timestamp(segment_end - 1),
            "first_input_timestamp": timestamp(first_start),
            "first_target_timestamp": timestamp(first_start + bundle.seq_length),
            "last_input_timestamp": timestamp(last_start + bundle.seq_length - 1),
            "last_target_timestamp": timestamp(
                last_start + bundle.seq_length + bundle.horizon - 1
            ),
        }
    return {
        "dataset": bundle.name,
        "time_points": int(bundle.raw.shape[0]),
        "nodes": int(bundle.raw.shape[1]),
        "train_windows": int(bundle.split_indices["train"].size),
        "validation_windows": int(bundle.split_indices["val"].size),
        "test_windows": int(bundle.split_indices["test"].size),
        "training_mean": bundle.mean,
        "training_std": bundle.std,
        "missing_or_zero_fraction": float((~bundle.valid).mean()),
        "adjacency_source": bundle.adjacency_source,
        "split": "raw-timeline chronological 70/10/20; windows built independently within each segment",
        "split_boundaries": boundaries,
        "raw_observation_overlap_across_splits": False,
        "input_steps": bundle.seq_length,
        "forecast_steps": bundle.horizon,
        "sampling_interval_minutes": 5,
    }
