#!/usr/bin/env python3
"""Stage 1: Strict patient-level StratifiedGroupKFold splits (zero patient leakage)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

DIR1_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIR1_ROOT / "src"))

from direction1.utils import (
    assert_no_patient_leakage,
    extract_tcga_patient_id,
    load_yaml_config,
    resolve_path,
)


def build_patient_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    """Create reproducible 5-fold StratifiedGroupKFold assignments grouped by 12-char patient_id."""
    if seeds is None:
        seeds = [42, 123, 2026]

    out = df.copy()
    if "slide_id" not in out.columns:
        raise KeyError("clinical.csv must contain a 'slide_id' column")

    if "patient_id" not in out.columns:
        out["patient_id"] = out["slide_id"].astype(str).map(extract_tcga_patient_id)
    else:
        out["patient_id"] = out["patient_id"].astype(str).map(extract_tcga_patient_id)

    # Sanity check: all slides of a patient must share the same OS label
    patient_label_counts = out.groupby("patient_id")[["os_time", "os_event"]].nunique()
    inconsistent = patient_label_counts[(patient_label_counts["os_time"] > 1) | (patient_label_counts["os_event"] > 1)]
    if not inconsistent.empty:
        raise ValueError(f"Inconsistent survival labels across slides for patients: {list(inconsistent.index)}")

    unique_patients = out["patient_id"].unique()
    if len(unique_patients) < n_splits:
        raise ValueError(f"Need at least {n_splits} unique patients for {n_splits}-fold CV, got {len(unique_patients)}")

    for seed in seeds:
        rng = np.random.default_rng(seed)
        shuffled_patients = unique_patients.copy()
        rng.shuffle(shuffled_patients)
        patient_order = {pid: rank for rank, pid in enumerate(shuffled_patients)}

        # Sort deterministically by shuffled patient rank so each seed produces distinct folds
        ordered = out.assign(_order=out["patient_id"].map(patient_order)).sort_values(["_order", "slide_id"])
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=False)

        fold_assign = pd.Series(index=ordered.index, dtype=np.int64)
        for fold_idx, (train_pos, val_pos) in enumerate(
            sgkf.split(ordered, ordered["os_event"].to_numpy(), groups=ordered["patient_id"].to_numpy())
        ):
            train_pids = ordered.iloc[train_pos]["patient_id"]
            val_pids = ordered.iloc[val_pos]["patient_id"]
            assert_no_patient_leakage(train_pids, val_pids, context=f"seed={seed}, fold={fold_idx}")
            fold_assign.iloc[val_pos] = fold_idx

        out[f"seed_{seed}_fold"] = fold_assign.reindex(out.index).astype(int)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: Patient-level StratifiedGroupKFold splits")
    parser.add_argument("--config", type=str, default=str(DIR1_ROOT / "configs/tcga_lihc_demo.yaml"))
    parser.add_argument("--clinical-csv", type=str, default=None, help="Override path to clinical.csv")
    parser.add_argument("--output-csv", type=str, default=None, help="Override path to output splits.csv")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic cohort paths")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    if args.synthetic:
        clinical_path = DIR1_ROOT / "data/synthetic/metadata/clinical.csv"
        output_path = DIR1_ROOT / "splits/synthetic_splits.csv"
    else:
        clinical_path = (
            Path(args.clinical_csv)
            if args.clinical_csv
            else resolve_path(DIR1_ROOT, cfg["paths"]["clinical_csv"])
        )
        output_path = (
            Path(args.output_csv)
            if args.output_csv
            else resolve_path(DIR1_ROOT, cfg["paths"]["splits_csv"])
        )

    df = pd.read_csv(clinical_path)
    n_splits = int(cfg["splits"]["n_splits"])
    seeds = [int(s) for s in cfg["splits"]["seeds"]]

    splits_df = build_patient_splits(df, n_splits=n_splits, seeds=seeds)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    splits_df.to_csv(output_path, index=False)

    print(
        f"[Stage 1] Saved {len(splits_df)} slides across {splits_df['patient_id'].nunique()} unique patients "
        f"-> {output_path}"
    )
    default_seed = int(cfg["splits"].get("default_seed", seeds[0]))
    col = f"seed_{default_seed}_fold"
    summary = (
        splits_df.groupby(col)
        .agg(
            n_slides=("slide_id", "count"),
            n_patients=("patient_id", "nunique"),
            events=("os_event", lambda s: int(splits_df.loc[s.index].groupby("patient_id")["os_event"].first().sum())),
        )
    )
    summary["event_rate"] = (summary["events"] / summary["n_patients"]).round(3)
    print(f"\nFold summary (seed {default_seed}):")
    print(summary.to_string())


if __name__ == "__main__":
    main()
