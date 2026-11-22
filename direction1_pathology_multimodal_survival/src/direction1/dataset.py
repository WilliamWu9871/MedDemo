"""Patient-level MIL dataset loading TRIDENT .h5 patch features and prior tokens."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from direction1.utils import encode_tsr_bins

TSR_BIN_COLS = [f"tsr_bin_{i}" for i in range(8)]


def _load_trident_h5(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read patch feature matrix (M, D) and coordinates (M, 2) from a TRIDENT .h5 file."""
    with h5py.File(h5_path, "r") as hf:
        if "features" in hf:
            feats = hf["features"][:]
        elif "feats" in hf:
            feats = hf["feats"][:]
        else:
            raise KeyError(f"Expected 'features' or 'feats' dataset in {h5_path}, found {list(hf.keys())}")
        coords = hf["coords"][:] if "coords" in hf else np.zeros((feats.shape[0], 2), dtype=np.int64)
    return np.asarray(feats, dtype=np.float32), np.asarray(coords, dtype=np.int64)


def build_h5_index(feats_dir: str | Path) -> tuple[dict[str, Path], dict[str, Path]]:
    """Index TRIDENT .h5 files by exact stem and unambiguous TCGA slide prefix."""
    root = Path(feats_dir)
    if not root.exists():
        return {}, {}
    exact_map: dict[str, Path] = {}
    prefix_buckets: dict[str, list[Path]] = {}
    for h5_path in sorted(root.rglob("*.h5")):
        exact_map[h5_path.stem] = h5_path
        prefix = h5_path.stem.split(".")[0]
        prefix_buckets.setdefault(prefix, []).append(h5_path)
    prefix_map = {prefix: paths[0] for prefix, paths in prefix_buckets.items() if len(paths) == 1}
    return exact_map, prefix_map


def resolve_h5_from_index(
    exact_map: dict[str, Path],
    prefix_map: dict[str, Path],
    slide_id: str,
) -> Path | None:
    """Resolve a slide_id to a TRIDENT .h5 path via exact stem or unique TCGA prefix."""
    if slide_id in exact_map:
        return exact_map[slide_id]
    return prefix_map.get(slide_id.split(".")[0])


def is_valid_trident_h5(h5_path: Path) -> bool:
    """Return True if h5_path is readable and contains a non-empty 2D feature matrix."""
    try:
        with h5py.File(h5_path, "r") as hf:
            key = "features" if "features" in hf else ("feats" if "feats" in hf else None)
            if key is None:
                return False
            ds = hf[key]
            return len(ds.shape) == 2 and ds.shape[0] > 0 and ds.shape[1] > 0
    except (OSError, KeyError, ValueError):
        return False


def preload_h5_cache(
    h5_paths: Iterable[Path],
    max_cache_gb: float = 8.0,
) -> tuple[dict[Path, tuple[np.ndarray, np.ndarray]], float]:
    """Preload TRIDENT .h5 features/coords into RAM once up to max_cache_gb."""
    cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
    max_bytes = int(max_cache_gb * (1024**3))
    total_bytes = 0
    for path in dict.fromkeys(h5_paths):
        feats, coords = _load_trident_h5(path)
        item_bytes = int(feats.nbytes + coords.nbytes)
        if cache and (total_bytes + item_bytes) > max_bytes:
            break
        cache[path] = (feats, coords)
        total_bytes += item_bytes
    return cache, total_bytes / (1024**3)


class PatientMILDataset(Dataset):
    """Bag dataset grouped by ``patient_id`` to prevent slide-to-patient leakage.

    Each item aggregates all diagnostic slides belonging to one patient into a single
    patch bag alongside the patient's TSR prior vector and standardized clinical vector.
    """

    def __init__(
        self,
        records_df: pd.DataFrame,
        feats_dir: str | Path,
        clinical_cols: list[str],
        tsr_num_bins: int = 8,
        max_patches: int | None = None,
        training: bool = True,
        seed: int = 42,
        h5_index: tuple[dict[str, Path], dict[str, Path]] | None = None,
        feature_cache: dict[Path, tuple[np.ndarray, np.ndarray]] | None = None,
    ):
        self.feats_dir = Path(feats_dir)
        self.clinical_cols = list(clinical_cols)
        self.tsr_num_bins = tsr_num_bins
        self.max_patches = max_patches
        self.training = training
        self.rng = np.random.default_rng(seed)
        self._exact_h5, self._prefix_h5 = h5_index if h5_index is not None else build_h5_index(self.feats_dir)
        self.feature_cache = feature_cache

        self.patients: list[dict[str, Any]] = []
        for patient_id, grp in records_df.groupby("patient_id", sort=False):
            slide_ids = grp["slide_id"].astype(str).tolist()
            h5_paths = [self._resolve_h5(sid) for sid in slide_ids]
            tsr_val = float(grp["pTSR"].mean())

            if all(col in grp.columns for col in TSR_BIN_COLS):
                tsr_bins = grp[TSR_BIN_COLS].iloc[0].to_numpy(dtype=np.float32)
            else:
                tsr_bins = np.asarray(encode_tsr_bins(tsr_val, num_bins=tsr_num_bins), dtype=np.float32)

            tsr_vec = np.concatenate([tsr_bins, np.asarray([tsr_val], dtype=np.float32)])
            clinical_vec = grp[self.clinical_cols].iloc[0].to_numpy(dtype=np.float32)

            os_time = float(grp["os_time"].iloc[0])
            os_event = int(grp["os_event"].iloc[0])
            y_disc = int(grp["y_disc"].iloc[0])
            censorship = 1 - os_event  # SurvPath convention: 1 = censored, 0 = death event

            self.patients.append(
                {
                    "patient_id": str(patient_id),
                    "slide_ids": slide_ids,
                    "h5_paths": h5_paths,
                    "tsr_vec": tsr_vec,
                    "pTSR": tsr_val,
                    "clinical_vec": clinical_vec,
                    "os_time": os_time,
                    "os_event": os_event,
                    "y_disc": y_disc,
                    "censorship": censorship,
                }
            )

        if not self.patients:
            raise ValueError("PatientMILDataset received 0 valid patients.")

        first_h5 = self.patients[0]["h5_paths"][0]
        if self.feature_cache is not None and first_h5 in self.feature_cache:
            sample_feats = self.feature_cache[first_h5][0]
        else:
            sample_feats, _ = _load_trident_h5(first_h5)
        self.feature_dim = int(sample_feats.shape[1])

    def _resolve_h5(self, slide_id: str) -> Path:
        resolved = resolve_h5_from_index(self._exact_h5, self._prefix_h5, slide_id)
        if resolved is not None:
            return resolved
        raise FileNotFoundError(f"Could not find feature .h5 for slide_id={slide_id!r} under {self.feats_dir}")

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.patients[idx]
        feats_list: list[np.ndarray] = []
        coords_list: list[np.ndarray] = []
        slide_offsets: list[tuple[str, int, int]] = []
        cursor = 0

        for slide_id, h5_path in zip(rec["slide_ids"], rec["h5_paths"]):
            if self.feature_cache is not None and h5_path in self.feature_cache:
                feats, coords = self.feature_cache[h5_path]
            else:
                feats, coords = _load_trident_h5(h5_path)
            n_patches = feats.shape[0]
            feats_list.append(feats)
            coords_list.append(coords)
            slide_offsets.append((slide_id, cursor, cursor + n_patches))
            cursor += n_patches

        if len(feats_list) == 1:
            bag_feats, bag_coords = feats_list[0], coords_list[0]
        else:
            bag_feats = np.concatenate(feats_list, axis=0)
            bag_coords = np.concatenate(coords_list, axis=0)

        if self.max_patches is not None and bag_feats.shape[0] > self.max_patches:
            if self.training:
                sel = self.rng.choice(bag_feats.shape[0], size=self.max_patches, replace=False)
                sel.sort()
            else:
                sel = np.linspace(0, bag_feats.shape[0] - 1, num=self.max_patches, dtype=np.int64)
            bag_feats = bag_feats[sel]
            bag_coords = bag_coords[sel]

        return {
            "patient_id": rec["patient_id"],
            "slide_ids": rec["slide_ids"],
            "slide_offsets": slide_offsets,
            "features": torch.from_numpy(bag_feats),
            "coords": torch.from_numpy(bag_coords),
            "tsr_vec": torch.from_numpy(rec["tsr_vec"]),
            "clinical_vec": torch.from_numpy(rec["clinical_vec"]),
            "y_disc": torch.tensor(rec["y_disc"], dtype=torch.long),
            "censorship": torch.tensor(rec["censorship"], dtype=torch.long),
            "os_time": rec["os_time"],
            "os_event": rec["os_event"],
            "pTSR": rec["pTSR"],
        }
