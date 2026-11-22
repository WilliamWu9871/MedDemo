"""Pure-Python utility helpers for Direction 1 patient IDs, TSR encoding, and split checks."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Iterable

_TCGA_PATIENT_RE = re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})(?:[-_.].*)?$", re.IGNORECASE)


def extract_tcga_patient_id(barcode: str) -> str:
    """Extract the 12-character TCGA patient barcode (e.g., TCGA-XX-XXXX).

    Falls back to stripping slide suffixes when given non-TCGA identifiers.
    """
    if not isinstance(barcode, str):
        raise TypeError(f"Expected string barcode, got {type(barcode).__name__}")
    cleaned = Path(barcode.strip()).name
    for ext in (".svs", ".h5", ".png", ".tif", ".tiff"):
        if cleaned.lower().endswith(ext):
            cleaned = cleaned[: -len(ext)]
            break

    match = _TCGA_PATIENT_RE.match(cleaned)
    if match:
        return match.group(1).upper()

    if "-slide" in cleaned.lower():
        idx = cleaned.lower().index("-slide")
        return cleaned[:idx]
    if "_" in cleaned:
        return cleaned.split("_", 1)[0]
    return cleaned


def encode_tsr_bins(tsr: float, num_bins: int = 8) -> list[float]:
    """Encode continuous TSR in [0, 1] into an 8-bin one-hot vector (Paper 1 Fig. 1F).

    Bins partition [0, 1] uniformly: [0, 0.125), [0.125, 0.25), ..., [0.875, 1.0].
    Values outside [0, 1] are clamped; exact 1.0 maps into the final bin.
    """
    if num_bins < 1:
        raise ValueError(f"num_bins must be >= 1, got {num_bins}")
    if math.isnan(tsr) or math.isinf(tsr):
        raise ValueError(f"tsr must be finite, got {tsr}")

    clamped = min(max(float(tsr), 0.0), 1.0)
    bin_idx = min(int(math.floor(clamped * num_bins)), num_bins - 1)
    one_hot = [0.0] * num_bins
    one_hot[bin_idx] = 1.0
    return one_hot


def assert_no_patient_leakage(
    train_patient_ids: Iterable[str],
    val_patient_ids: Iterable[str],
    context: str = "",
) -> None:
    """Raise ValueError if any patient_id appears in both train and validation sets."""
    train_set = set(train_patient_ids)
    val_set = set(val_patient_ids)
    overlap = sorted(train_set.intersection(val_set))
    if overlap:
        prefix = f"[{context}] " if context else ""
        preview = ", ".join(overlap[:10])
        raise ValueError(
            f"{prefix}Patient-level data leakage detected! "
            f"{len(overlap)} overlapping patient(s): {preview}"
        )


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and return a dictionary."""
    import yaml

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(data).__name__}")
    return data


def resolve_path(base_dir: str | Path, path_value: str | Path) -> Path:
    """Resolve a relative path against base_dir, keeping absolute paths unchanged."""
    p = Path(path_value)
    return p if p.is_absolute() else (Path(base_dir) / p).resolve()
