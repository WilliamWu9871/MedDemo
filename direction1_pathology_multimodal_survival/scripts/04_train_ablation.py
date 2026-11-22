#!/usr/bin/env python3
"""Stage 3B: Strictly aligned 5-fold ablation across Exp 1.0, Exp 1.1, and Exp 1.2.

Models evaluated on identical patient-level StratifiedGroupKFold splits:
  - exp1_0_clinical_cox:          Clinical-only CoxPH baseline (lifelines)
  - exp1_0b_clinical_tsr_cox:     Clinical + pTSR CoxPH comparator
  - exp1_1_wsi_abmil:             WSI-only Gated ABMIL (Discrete NLLSurvLoss)
  - exp1_2a_wsi_tsr_token:        Token-Guided Cross-Attention MIL (WSI + TSR Token)
  - exp1_2b_wsi_tsr_clinical_token: Token-Guided Cross-Attention MIL (WSI + TSR + Clinical Tokens)
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

DIR1_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIR1_ROOT / "src"))

from direction1.dataset import PatientMILDataset
from direction1.losses import NLLSurvLoss, fit_survival_bins, transform_survival_bins
from direction1.models import GatedABMIL, TokenGuidedMultimodalSurv
from direction1.utils import assert_no_patient_leakage, load_yaml_config, resolve_path


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _safe_cindex(times: np.ndarray, risks: np.ndarray, events: np.ndarray) -> float:
    if len(times) < 2 or int(np.sum(events)) == 0:
        return float("nan")
    try:
        # lifelines concordance_index expects higher predicted scores -> longer survival -> use -risk
        return float(concordance_index(times, -np.asarray(risks, dtype=np.float64), event_observed=events))
    except ZeroDivisionError:
        return float("nan")


def _run_cox_fold(
    train_pat: pd.DataFrame,
    val_pat: pd.DataFrame,
    feature_cols: list[str],
    exp_name: str,
    seed: int,
    fold: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    cols = list(dict.fromkeys(feature_cols))
    tr_df = train_pat[cols + ["os_time", "os_event"]].copy()
    va_df = val_pat[cols + ["os_time", "os_event"]].copy()

    # Drop constant columns in training fold to keep Cox optimization well-conditioned
    active_cols = [c for c in cols if tr_df[c].std(ddof=0) > 1e-8]
    if not active_cols:
        tr_risk = np.zeros(len(tr_df), dtype=np.float64)
        va_risk = np.zeros(len(va_df), dtype=np.float64)
    else:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(tr_df[active_cols + ["os_time", "os_event"]], duration_col="os_time", event_col="os_event")
        tr_risk = np.log(cph.predict_partial_hazard(tr_df[active_cols]).to_numpy().ravel() + 1e-12)
        va_risk = np.log(cph.predict_partial_hazard(va_df[active_cols]).to_numpy().ravel() + 1e-12)

    tr_c = _safe_cindex(tr_df["os_time"].to_numpy(), tr_risk, tr_df["os_event"].to_numpy())
    va_c = _safe_cindex(va_df["os_time"].to_numpy(), va_risk, va_df["os_event"].to_numpy())

    metric_row = {
        "seed": seed,
        "fold": fold,
        "exp_name": exp_name,
        "train_cindex": tr_c,
        "val_cindex": va_c,
        "n_train_patients": len(tr_df),
        "n_val_patients": len(va_df),
    }
    oof_cols = list(dict.fromkeys(["patient_id", "os_time", "os_event", "pTSR"] + cols))
    oof_df = val_pat[oof_cols].copy().reset_index(drop=True)
    oof_df["seed"] = seed
    oof_df["fold"] = fold
    oof_df["exp_name"] = exp_name
    oof_df["risk_score"] = va_risk
    return metric_row, oof_df


def _mil_collate_single(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return batch[0]


def _evaluate_mil(
    model: torch.nn.Module,
    dataset: PatientMILDataset,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=_mil_collate_single)
    times, events, risks = [], [], []
    slide_attns: list[dict[str, Any]] = []

    with torch.no_grad():
        for item in loader:
            feats = item["features"].to(device)
            tsr_vec = item["tsr_vec"].to(device)
            clin_vec = item["clinical_vec"].to(device)
            out = model(feats, tsr_vec=tsr_vec, clinical_vec=clin_vec)
            risk_val = float(out["risk"].item())
            times.append(item["os_time"])
            events.append(item["os_event"])
            risks.append(risk_val)

            attn = out["attn_weights"].detach().cpu().numpy()
            coords = item["coords"].numpy()
            for slide_id, start, end in item["slide_offsets"]:
                if end > start and end <= len(attn):
                    s_attn = attn[start:end]
                    s_coords = coords[start:end]
                else:
                    s_attn = attn
                    s_coords = coords
                s_sum = float(np.sum(s_attn))
                if s_sum > 0:
                    s_attn = s_attn / s_sum
                slide_attns.append(
                    {
                        "patient_id": item["patient_id"],
                        "slide_id": slide_id,
                        "coords": s_coords,
                        "attn_weights": s_attn.astype(np.float32),
                        "risk_score": risk_val,
                        "pTSR": item["pTSR"],
                    }
                )

    t_arr = np.asarray(times, dtype=np.float64)
    e_arr = np.asarray(events, dtype=np.int64)
    r_arr = np.asarray(risks, dtype=np.float64)
    return _safe_cindex(t_arr, r_arr, e_arr), t_arr, r_arr, e_arr, slide_attns


def _build_dl_model(
    exp_name: str,
    in_dim: int,
    clinical_dim: int,
    cfg: dict[str, Any],
) -> torch.nn.Module:
    m_cfg = cfg["model"]
    hidden_dim = int(m_cfg.get("hidden_dim", 256))
    n_bins = int(m_cfg.get("n_bins", 4))
    n_heads = int(m_cfg.get("n_heads", 4))
    dropout = float(m_cfg.get("dropout", 0.25))
    tsr_dim = int(cfg["prior"].get("tsr_num_bins", 8)) + 1

    if exp_name == "exp1_1_wsi_abmil":
        return GatedABMIL(in_dim=in_dim, hidden_dim=hidden_dim, n_bins=n_bins, dropout=dropout)
    if exp_name == "exp1_2a_wsi_tsr_token":
        return TokenGuidedMultimodalSurv(
            in_dim=in_dim,
            tsr_dim=tsr_dim,
            clinical_dim=clinical_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_bins=n_bins,
            dropout=dropout,
            use_clinical=False,
        )
    if exp_name == "exp1_2b_wsi_tsr_clinical_token":
        return TokenGuidedMultimodalSurv(
            in_dim=in_dim,
            tsr_dim=tsr_dim,
            clinical_dim=clinical_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_bins=n_bins,
            dropout=dropout,
            use_clinical=True,
        )
    raise ValueError(f"Unknown DL experiment: {exp_name}")


def _run_dl_fold(
    exp_name: str,
    train_ds: PatientMILDataset,
    val_ds: PatientMILDataset,
    val_pat_df: pd.DataFrame,
    clinical_cols: list[str],
    cfg: dict[str, Any],
    seed: int,
    fold: int,
    epochs_override: int | None,
    device: torch.device,
    attention_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    t_cfg = cfg["training"]
    epochs = int(epochs_override or t_cfg.get("epochs", 25))
    grad_accum = int(t_cfg.get("grad_accum_steps", 16))
    lr = float(t_cfg.get("lr", 2e-4))
    weight_decay = float(t_cfg.get("weight_decay", 1e-4))
    alpha = float(t_cfg.get("nll_alpha", 0.15))
    patience = int(t_cfg.get("early_stop_patience", 8))

    _set_seed(seed * 100 + fold)
    model = _build_dl_model(
        exp_name=exp_name,
        in_dim=train_ds.feature_dim,
        clinical_dim=len(clinical_cols),
        cfg=cfg,
    ).to(device)

    criterion = NLLSurvLoss(alpha=alpha, reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=_mil_collate_single)

    best_val_c = -1.0
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_in_accum = 0

        for idx, item in enumerate(train_loader):
            feats = item["features"].to(device)
            tsr_vec = item["tsr_vec"].to(device)
            clin_vec = item["clinical_vec"].to(device)
            y_disc = item["y_disc"].to(device).view(1)
            censorship = item["censorship"].to(device).view(1)

            out = model(feats, tsr_vec=tsr_vec, clinical_vec=clin_vec)
            loss = criterion(out["logits"], y_disc, censorship)
            (loss / grad_accum).backward()
            step_in_accum += 1

            if step_in_accum >= grad_accum or (idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step_in_accum = 0

        scheduler.step()
        val_c, _, _, _, _ = _evaluate_mil(model, val_ds, device)
        if not np.isnan(val_c) and val_c > best_val_c:
            best_val_c = val_c
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if epoch >= 5 and no_improve >= patience:
                break

    model.load_state_dict(best_state)
    tr_c, _, _, _, _ = _evaluate_mil(model, train_ds, device)
    va_c, _, va_risks, _, val_slide_attns = _evaluate_mil(model, val_ds, device)

    exp_attn_dir = attention_dir / f"seed_{seed}" / exp_name
    exp_attn_dir.mkdir(parents=True, exist_ok=True)
    for rec in val_slide_attns:
        np.savez_compressed(
            exp_attn_dir / f"{rec['slide_id']}.npz",
            coords=rec["coords"],
            attn_weights=rec["attn_weights"],
            risk_score=np.asarray([rec["risk_score"]], dtype=np.float32),
            pTSR=np.asarray([rec["pTSR"]], dtype=np.float32),
        )

    metric_row = {
        "seed": seed,
        "fold": fold,
        "exp_name": exp_name,
        "train_cindex": tr_c,
        "val_cindex": va_c,
        "n_train_patients": len(train_ds),
        "n_val_patients": len(val_ds),
    }

    oof_cols = list(dict.fromkeys(["patient_id", "os_time", "os_event", "pTSR"] + clinical_cols))
    oof_df = val_pat_df[oof_cols].copy().reset_index(drop=True)
    risk_by_pat = {p["patient_id"]: float(r) for p, r in zip(val_ds.patients, va_risks)}
    oof_df["seed"] = seed
    oof_df["fold"] = fold
    oof_df["exp_name"] = exp_name
    oof_df["risk_score"] = oof_df["patient_id"].map(risk_by_pat)
    return metric_row, oof_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3B: Run aligned 5-fold MIL + Prior Token ablations")
    parser.add_argument("--config", type=str, default=str(DIR1_ROOT / "configs/tcga_lihc_demo.yaml"))
    parser.add_argument("--splits-csv", type=str, default=None)
    parser.add_argument("--priors-csv", type=str, default=None)
    parser.add_argument("--feats-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--attention-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--all-seeds", action="store_true", help="Run all seeds defined in config")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic cohort outputs")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    if args.synthetic:
        splits_path = DIR1_ROOT / "splits/synthetic_splits.csv"
        priors_path = DIR1_ROOT / "outputs/synthetic/priors/slide_priors.csv"
        feats_dir = DIR1_ROOT / "data/synthetic/trident_feats/features"
        out_dir = DIR1_ROOT / "outputs/synthetic/ablation"
        attn_dir = DIR1_ROOT / "outputs/synthetic/attention"
    else:
        splits_path = Path(args.splits_csv) if args.splits_csv else resolve_path(DIR1_ROOT, cfg["paths"]["splits_csv"])
        priors_path = Path(args.priors_csv) if args.priors_csv else resolve_path(DIR1_ROOT, cfg["paths"]["priors_csv"])
        feats_dir = (
            Path(args.feats_dir)
            if args.feats_dir
            else resolve_path(DIR1_ROOT, cfg["paths"]["trident_job_dir"])
        )
        out_dir = Path(args.output_dir) if args.output_dir else resolve_path(DIR1_ROOT, cfg["paths"]["ablation_dir"])
        attn_dir = Path(args.attention_dir) if args.attention_dir else resolve_path(DIR1_ROOT, cfg["paths"]["attention_dir"])

    out_dir.mkdir(parents=True, exist_ok=True)
    attn_dir.mkdir(parents=True, exist_ok=True)

    splits_df = pd.read_csv(splits_path)
    priors_df = pd.read_csv(priors_path)
    tsr_cols = [c for c in priors_df.columns if c == "pTSR" or c.startswith("tsr_bin_")]
    merged = splits_df.merge(priors_df[["slide_id"] + tsr_cols], on="slide_id", how="left")

    clinical_cols = list(cfg["data"]["clinical_cols"])
    for col in clinical_cols:
        if col not in merged.columns:
            raise KeyError(f"Clinical column {col!r} missing from splits/clinical table.")
        if merged[col].isna().any():
            merged[col] = merged[col].fillna(merged[col].median())

    seeds = (
        [int(s) for s in cfg["splits"]["seeds"]]
        if args.all_seeds
        else [int(args.seed or cfg["splits"].get("default_seed", 42))]
    )
    n_splits = int(cfg["splits"]["n_splits"])
    n_bins = int(cfg["model"].get("n_bins", 4))
    tsr_num_bins = int(cfg["prior"].get("tsr_num_bins", 8))
    max_patches_train = cfg["data"].get("max_patches_train")
    max_patches_eval = cfg["data"].get("max_patches_eval")

    device = _select_device()
    print(f"[Stage 3B] Device: {device} | Seeds: {seeds} | Clinical cols: {clinical_cols}")

    all_metrics: list[dict[str, Any]] = []
    all_oofs: list[pd.DataFrame] = []

    dl_exps = [
        "exp1_1_wsi_abmil",
        "exp1_2a_wsi_tsr_token",
        "exp1_2b_wsi_tsr_clinical_token",
    ]

    for seed in seeds:
        fold_col = f"seed_{seed}_fold"
        if fold_col not in merged.columns:
            raise KeyError(f"Column {fold_col!r} not found in {splits_path}")

        for fold in range(n_splits):
            tr_slides = merged[merged[fold_col] != fold].copy()
            va_slides = merged[merged[fold_col] == fold].copy()
            assert_no_patient_leakage(
                tr_slides["patient_id"], va_slides["patient_id"], context=f"seed={seed}, fold={fold}"
            )

            # Patient-level table for fitting clinical scalers, discrete bins, and CoxPH
            tr_pat = tr_slides.groupby("patient_id", as_index=False).agg(
                {
                    "os_time": "first",
                    "os_event": "first",
                    "pTSR": "mean",
                    **{c: "first" for c in clinical_cols},
                }
            )
            va_pat = va_slides.groupby("patient_id", as_index=False).agg(
                {
                    "os_time": "first",
                    "os_event": "first",
                    "pTSR": "mean",
                    **{c: "first" for c in clinical_cols},
                }
            )

            # Standardize clinical features strictly on training fold
            scaler = StandardScaler()
            tr_clin_scaled = scaler.fit_transform(tr_pat[clinical_cols].to_numpy(dtype=np.float32))
            va_clin_scaled = scaler.transform(va_pat[clinical_cols].to_numpy(dtype=np.float32))
            tr_pat_std = tr_pat.copy()
            va_pat_std = va_pat.copy()
            for idx_c, col in enumerate(clinical_cols):
                tr_pat_std[col] = tr_clin_scaled[:, idx_c]
                va_pat_std[col] = va_clin_scaled[:, idx_c]

            # Propagate fold-standardized clinical columns back to slide rows for MIL datasets
            clin_tr_map = tr_pat_std.set_index("patient_id")[clinical_cols].to_dict("index")
            clin_va_map = va_pat_std.set_index("patient_id")[clinical_cols].to_dict("index")
            for col in clinical_cols:
                tr_slides[col] = tr_slides["patient_id"].map(lambda pid: clin_tr_map[pid][col])
                va_slides[col] = va_slides["patient_id"].map(lambda pid: clin_va_map[pid][col])

            # Fit 4-bin discrete survival cutpoints strictly on uncensored train patients
            bin_edges = fit_survival_bins(
                tr_pat["os_time"].to_numpy(),
                tr_pat["os_event"].to_numpy(),
                n_bins=n_bins,
            )
            tr_slides["y_disc"] = transform_survival_bins(tr_slides["os_time"].to_numpy(), bin_edges)
            va_slides["y_disc"] = transform_survival_bins(va_slides["os_time"].to_numpy(), bin_edges)

            # Exp 1.0: Clinical-only CoxPH & Clinical+TSR CoxPH
            m0, oof0 = _run_cox_fold(tr_pat_std, va_pat_std, clinical_cols, "exp1_0_clinical_cox", seed, fold)
            m0b, oof0b = _run_cox_fold(
                tr_pat_std, va_pat_std, clinical_cols + ["pTSR"], "exp1_0b_clinical_tsr_cox", seed, fold
            )
            all_metrics.extend([m0, m0b])
            all_oofs.extend([oof0, oof0b])

            train_ds = PatientMILDataset(
                tr_slides,
                feats_dir=feats_dir,
                clinical_cols=clinical_cols,
                tsr_num_bins=tsr_num_bins,
                max_patches=max_patches_train,
                training=True,
                seed=seed + fold,
            )
            val_ds = PatientMILDataset(
                va_slides,
                feats_dir=feats_dir,
                clinical_cols=clinical_cols,
                tsr_num_bins=tsr_num_bins,
                max_patches=max_patches_eval,
                training=False,
                seed=seed + fold,
            )

            for exp_name in dl_exps:
                m_dl, oof_dl = _run_dl_fold(
                    exp_name=exp_name,
                    train_ds=train_ds,
                    val_ds=val_ds,
                    val_pat_df=va_pat,
                    clinical_cols=clinical_cols,
                    cfg=cfg,
                    seed=seed,
                    fold=fold,
                    epochs_override=args.epochs,
                    device=device,
                    attention_dir=attn_dir,
                )
                all_metrics.append(m_dl)
                all_oofs.append(oof_dl)

            print(
                f"  [seed={seed} fold={fold}] "
                + " | ".join(
                    f"{row['exp_name'].split('_', 2)[-1]}={row['val_cindex']:.3f}"
                    for row in all_metrics[-5:]
                )
            )

    metrics_df = pd.DataFrame(all_metrics)
    oofs_df = pd.concat(all_oofs, ignore_index=True)
    metrics_df.to_csv(out_dir / "fold_metrics.csv", index=False)
    oofs_df.to_csv(out_dir / "oof_predictions.csv", index=False)

    summary = metrics_df.groupby("exp_name")[["train_cindex", "val_cindex"]].agg(["mean", "std"]).round(4)
    print(f"\n[Stage 3B] Saved results -> {out_dir}")
    print(summary.to_string())


if __name__ == "__main__":
    main()
