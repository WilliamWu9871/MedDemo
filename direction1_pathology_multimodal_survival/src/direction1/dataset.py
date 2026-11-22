"""Patient-level MIL dataset loading TRIDENT .h5 patch features and prior tokens."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
    ):
        self.feats_dir = Path(feats_dir)
        self.clinical_cols = list(clinical_cols)
        self.tsr_num_bins = tsr_num_bins
        self.max_patches = max_patches
        self.training = training
        self.rng = np.random.default_rng(seed)

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

        sample_feats, _ = _load_trident_h5(self.patients[0]["h5_paths"][0])
        self.feature_dim = int(sample_feats.shape[1])

    def _resolve_h5(self, slide_id: str) -> Path:
        direct = self.feats_dir / f"{slide_id}.h5"
        if direct.exists():
            return direct
        matches = list(self.feats_dir.rglob(f"{slide_id}.h5"))
        if matches:
            return matches[0]
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
            feats, coords = _load_trident_h5(h5_path)
            n_patches = feats.shape[0]
            feats_list.append(feats)
            coords_list.append(coords)
            slide_offsets.append((slide_id, cursor, cursor + n_patches))
            cursor += n_patches

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
