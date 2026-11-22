#!/usr/bin/env python3
"""Stage 4: Clinical survival statistics, RCS non-linearity check, SHAP, and WSI attention overlays.

Generates all 4+ acceptance deliverables under outputs/.../report/:
  1. 5-fold + pooled OOF C-index & time-dependent AUC table + leakage audit
  2. Median-split Kaplan-Meier survival curves + Log-rank p-values
  3. Multivariable Cox regression (AI Risk Score + clinical covariates)
  4. Restricted Cubic Spline (RCS) HR curve vs pTSR (4 knots at 5th/35th/65th/95th pct)
  5. SHAP / permutation feature contribution plot
  6. WSI Attention Heatmap overlays (Exp 1.1 ABMIL vs Exp 1.2b Token-Guided) + background noise audit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from PIL import Image
from sklearn.linear_model import Ridge

DIR1_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIR1_ROOT / "src"))

from direction1.utils import load_yaml_config, resolve_path

EXP_DISPLAY = {
    "exp1_0_clinical_cox": "Exp 1.0 Clinical CoxPH",
    "exp1_0b_clinical_tsr_cox": "Exp 1.0b Clinical+TSR CoxPH",
    "exp1_1_wsi_abmil": "Exp 1.1 WSI-only ABMIL",
    "exp1_2a_wsi_tsr_token": "Exp 1.2a WSI+TSR Token",
    "exp1_2b_wsi_tsr_clinical_token": "Exp 1.2b WSI+TSR+Clinical Token",
}


def _compute_ipcw_auc(oof_df: pd.DataFrame, horizon_days: float = 1095.0) -> float:
    try:
        from sksurv.metrics import cumulative_dynamic_auc
        from sksurv.util import Surv
    except ImportError:
        return float("nan")

    y_struct = Surv.from_arrays(
        event=oof_df["os_event"].astype(bool).to_numpy(),
        time=oof_df["os_time"].astype(float).to_numpy(),
    )
    max_event_t = float(oof_df.loc[oof_df["os_event"] == 1, "os_time"].max())
    min_t = float(oof_df["os_time"].min())
    eval_t = min(horizon_days, max_event_t * 0.90)
    if eval_t <= min_t or np.isnan(eval_t):
        return float("nan")
    try:
        aucs, _ = cumulative_dynamic_auc(
            y_struct,
            y_struct,
            oof_df["risk_score"].to_numpy(dtype=np.float64),
            times=np.asarray([eval_t]),
        )
        return float(aucs[0])
    except Exception:
        return float("nan")


def build_cindex_report(fold_df: pd.DataFrame, oof_df: pd.DataFrame, report_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for exp_name, grp in fold_df.groupby("exp_name", sort=False):
        oof_sub = oof_df[oof_df["exp_name"] == exp_name]
        # Aggregate per patient across seeds if multiple seeds exist
        pat_oof = oof_sub.groupby("patient_id", as_index=False).agg(
            {"os_time": "first", "os_event": "first", "risk_score": "mean"}
        )
        try:
            pooled_c = float(
                concordance_index(
                    pat_oof["os_time"],
                    -pat_oof["risk_score"],
                    event_observed=pat_oof["os_event"],
                )
            )
        except ZeroDivisionError:
            pooled_c = float("nan")
        tr_mean = float(grp["train_cindex"].mean())
        va_mean = float(grp["val_cindex"].mean())
        va_std = float(grp["val_cindex"].std(ddof=1)) if len(grp) > 1 else 0.0
        auc_3y = _compute_ipcw_auc(pat_oof, horizon_days=1095.0)
        leakage_flag = bool(tr_mean > 0.92 and va_mean < 0.55)
        rows.append(
            {
                "exp_name": exp_name,
                "label": EXP_DISPLAY.get(exp_name, exp_name),
                "train_cindex_mean": round(tr_mean, 4),
                "val_cindex_mean": round(va_mean, 4),
                "val_cindex_std": round(va_std, 4),
                "pooled_oof_cindex": round(pooled_c, 4),
                "ipcw_auc_3yr": round(auc_3y, 4) if not np.isnan(auc_3y) else np.nan,
                "leakage_red_line": leakage_flag,
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(report_dir / "ablation_cindex_table.csv", index=False)

    md_lines = [
        "| Model | Train C-index | Val C-index (mean ± std) | Pooled OOF C-index | 3-Yr IPCW AUC | Red-Line |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        auc_str = f"{r['ipcw_auc_3yr']:.3f}" if pd.notna(r["ipcw_auc_3yr"]) else "N/A"
        flag_str = "RED-LINE: OVERFIT/LEAK" if r["leakage_red_line"] else "OK"
        md_lines.append(
            f"| {r['label']} | {r['train_cindex_mean']:.3f} | "
            f"{r['val_cindex_mean']:.3f} ± {r['val_cindex_std']:.3f} | "
            f"{r['pooled_oof_cindex']:.3f} | {auc_str} | {flag_str} |"
        )
    (report_dir / "ablation_cindex_table.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    for r in rows:
        if r["leakage_red_line"]:
            print(
                f"[RED-LINE WARNING] {r['label']}: train C-index={r['train_cindex_mean']:.3f} "
                f"vs val C-index={r['val_cindex_mean']:.3f}. Check patient splits or regularization!"
            )
    return table


def plot_km_curves(oof_df: pd.DataFrame, report_dir: Path) -> None:
    exps = [e for e in EXP_DISPLAY if e in set(oof_df["exp_name"])]
    n_panels = len(exps)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.2), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, exp_name in zip(axes, exps):
        sub = (
            oof_df[oof_df["exp_name"] == exp_name]
            .groupby("patient_id", as_index=False)
            .agg({"os_time": "first", "os_event": "first", "risk_score": "mean"})
        )
        med = float(sub["risk_score"].median())
        high = sub[sub["risk_score"] > med]
        low = sub[sub["risk_score"] <= med]
        if high.empty or low.empty:
            med = float(sub["risk_score"].quantile(0.5))
            high = sub.iloc[: len(sub) // 2]
            low = sub.iloc[len(sub) // 2 :]

        res = logrank_test(
            high["os_time"], low["os_time"], event_observed_A=high["os_event"], event_observed_B=low["os_event"]
        )
        kmf_low = KaplanMeierFitter()
        kmf_high = KaplanMeierFitter()
        kmf_low.fit(low["os_time"], event_observed=low["os_event"], label=f"Low-Risk (n={len(low)})")
        kmf_high.fit(high["os_time"], event_observed=high["os_event"], label=f"High-Risk (n={len(high)})")
        kmf_low.plot_survival_function(ax=ax, ci_show=True, color="#1f77b4")
        kmf_high.plot_survival_function(ax=ax, ci_show=True, color="#d62728")

        ax.set_title(f"{EXP_DISPLAY.get(exp_name, exp_name)}\nLog-rank p = {res.p_value:.4g}", fontsize=10)
        ax.set_xlabel("Overall Survival Time (days)")
        if ax is axes[0]:
            ax.set_ylabel("Survival Probability")
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(report_dir / "km_survival_curves.png", dpi=200)
    plt.close(fig)


def run_multivariable_cox(
    oof_df: pd.DataFrame,
    clinical_cols: list[str],
    target_exp: str,
    report_dir: Path,
) -> pd.DataFrame:
    sub = oof_df[oof_df["exp_name"] == target_exp]
    if sub.empty:
        sub = oof_df[oof_df["exp_name"] == oof_df["exp_name"].iloc[0]]

    pat = sub.groupby("patient_id", as_index=False).agg(
        {
            "os_time": "first",
            "os_event": "first",
            "risk_score": "mean",
            "pTSR": "mean",
            **{c: "first" for c in clinical_cols},
        }
    )
    cox_df = pd.DataFrame(
        {
            "AI_Risk_Score_Z": (pat["risk_score"] - pat["risk_score"].mean()) / (pat["risk_score"].std() + 1e-8),
            "os_time": pat["os_time"],
            "os_event": pat["os_event"],
        }
    )
    for col in clinical_cols:
        if pat[col].std(ddof=0) > 1e-8:
            cox_df[col] = (pat[col] - pat[col].mean()) / pat[col].std()

    cph = CoxPHFitter(penalizer=0.05)
    cph.fit(cox_df, duration_col="os_time", event_col="os_event")
    summary = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].copy()
    summary.columns = ["HR", "CI_95_lower", "CI_95_upper", "p_value"]
    summary.to_csv(report_dir / "multivariable_cox.csv")

    fig, ax = plt.subplots(figsize=(6.5, max(2.8, 0.7 * len(summary) + 1.2)))
    cph.plot(ax=ax)
    ax.set_title(f"Multivariable Cox ({EXP_DISPLAY.get(target_exp, target_exp)})")
    fig.tight_layout()
    fig.savefig(report_dir / "multivariable_cox_forest.png", dpi=200)
    plt.close(fig)
    return summary


def plot_rcs_tsr_curve(oof_df: pd.DataFrame, knots_pct: list[float], report_dir: Path) -> None:
    """Fit a 4-knot Restricted Cubic Spline Cox model on pTSR (Paper 1 Fig. 3A check)."""
    from patsy import dmatrix

    pat = (
        oof_df.groupby("patient_id", as_index=False)
        .agg({"os_time": "first", "os_event": "first", "pTSR": "mean"})
        .dropna()
    )
    tsr_vals = pat["pTSR"].to_numpy(dtype=np.float64)
    if np.std(tsr_vals) < 1e-5:
        return

    knots = np.quantile(tsr_vals, [p / 100.0 for p in knots_pct])
    lower, upper = float(knots[0]), float(knots[-1])
    inner = [float(k) for k in knots[1:-1]]

    formula = (
        f"cr(pTSR, knots={inner}, lower_bound={lower}, upper_bound={upper}) - 1"
    )
    basis = np.asarray(dmatrix(formula, {"pTSR": tsr_vals}))
    rcs_cols = [f"rcs_{i}" for i in range(basis.shape[1])]
    fit_df = pd.DataFrame(basis, columns=rcs_cols)
    fit_df["os_time"] = pat["os_time"].to_numpy()
    fit_df["os_event"] = pat["os_event"].to_numpy()

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(fit_df, duration_col="os_time", event_col="os_event")

    grid_tsr = np.linspace(float(np.quantile(tsr_vals, 0.03)), float(np.quantile(tsr_vals, 0.97)), 120)
    grid_basis = np.asarray(dmatrix(formula, {"pTSR": grid_tsr}))
    ref_tsr = float(np.median(tsr_vals))
    ref_basis = np.asarray(dmatrix(formula, {"pTSR": np.array([ref_tsr])}))
    diff_basis = grid_basis - ref_basis

    beta = cph.params_.to_numpy()
    cov = cph.variance_matrix_.to_numpy()
    log_hr = diff_basis @ beta
    se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", diff_basis, cov, diff_basis), 1e-12))
    hr = np.exp(log_hr)
    hr_lo = np.exp(log_hr - 1.96 * se)
    hr_hi = np.exp(log_hr + 1.96 * se)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(grid_tsr, hr, color="#7b2cbf", lw=2.2, label="Estimated HR")
    ax.fill_between(grid_tsr, hr_lo, hr_hi, color="#c77dff", alpha=0.28, label="95% CI")
    ax.axhline(1.0, color="gray", linestyle="--", lw=1.0)
    peak_tsr = float(grid_tsr[int(np.argmax(hr))])
    ax.axvline(peak_tsr, color="#e63946", linestyle=":", lw=1.2, label=f"Peak HR @ TSR={peak_tsr:.3f}")
    ax.set_xlabel("Predicted Tumor-Stroma Ratio (pTSR)")
    ax.set_ylabel("Hazard Ratio (vs median TSR)")
    ax.set_title("Restricted Cubic Spline (4 knots) of TSR vs OS")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(report_dir / "rcs_tsr_curve.png", dpi=200)
    plt.close(fig)


def plot_shap_summary(
    oof_df: pd.DataFrame,
    clinical_cols: list[str],
    target_exp: str,
    report_dir: Path,
) -> None:
    """Compute SHAP feature contributions for [pTSR + clinical covariates] explaining AI Risk Score."""
    sub = oof_df[oof_df["exp_name"] == target_exp]
    if sub.empty:
        return
    pat = sub.groupby("patient_id", as_index=False).agg(
        {"risk_score": "mean", "pTSR": "mean", **{c: "first" for c in clinical_cols}}
    )
    feat_cols = ["pTSR"] + clinical_cols
    X = pat[feat_cols].copy()
    X = (X - X.mean()) / (X.std() + 1e-8)
    y = pat["risk_score"].to_numpy(dtype=np.float64)

    surrogate = Ridge(alpha=1.0).fit(X, y)
    try:
        import shap

        explainer = shap.LinearExplainer(surrogate, X)
        shap_vals = explainer.shap_values(X)
        plt.figure(figsize=(6.5, 3.8))
        shap.summary_plot(shap_vals, X, show=False)
        plt.title(f"SHAP Feature Importance ({EXP_DISPLAY.get(target_exp, target_exp)})")
        plt.tight_layout()
        plt.savefig(report_dir / "shap_summary.png", dpi=200, bbox_inches="tight")
        plt.close()
    except Exception:
        importances = np.abs(surrogate.coef_)
        fig, ax = plt.subplots(figsize=(6.0, 3.5))
        ax.barh(feat_cols, importances, color="#457b9d")
        ax.set_xlabel("|Standardized Effect on AI Risk Score|")
        ax.set_title(f"Feature Importance ({EXP_DISPLAY.get(target_exp, target_exp)})")
        fig.tight_layout()
        fig.savefig(report_dir / "shap_summary.png", dpi=200)
        plt.close(fig)


def _load_thumb_rgb(
    slide_id: str,
    wsi_dir: Path | None,
    thumbs_dir: Path | None,
    max_dim: int = 512,
) -> tuple[np.ndarray | None, tuple[int, int] | None]:
    dims0: tuple[int, int] | None = None
    slide_obj = None
    if wsi_dir is not None and wsi_dir.exists():
        matches = list(wsi_dir.glob(f"{slide_id}*.svs"))
        if matches:
            import openslide

            slide_obj = openslide.OpenSlide(str(matches[0]))
            dims0 = slide_obj.dimensions

    if thumbs_dir is not None and thumbs_dir.exists():
        for ext in (".png", ".jpg", ".jpeg"):
            cand = thumbs_dir / f"{slide_id}{ext}"
            if cand.exists():
                img = np.asarray(Image.open(cand).convert("RGB"))
                if dims0 is None:
                    dims0 = (img.shape[1] * 16, img.shape[0] * 16)
                return img, dims0

    if slide_obj is not None and dims0 is not None:
        thumb = np.asarray(slide_obj.get_thumbnail((max_dim, max_dim)).convert("RGB"))
        return thumb, dims0
    return None, None


def render_attention_heatmaps(
    oof_df: pd.DataFrame,
    priors_df: pd.DataFrame,
    attn_dir: Path,
    wsi_dir: Path | None,
    thumbs_dir: Path | None,
    seed: int,
    n_slides: int,
    bg_rgb_thresh: int,
    bg_warn_frac: float,
    report_dir: Path,
) -> None:
    """Overlay Exp 1.1 ABMIL vs Exp 1.2b Token-Guided attention onto WSI thumbnails."""
    heat_dir = report_dir / "heatmaps"
    heat_dir.mkdir(parents=True, exist_ok=True)

    abmil_dir = attn_dir / f"seed_{seed}" / "exp1_1_wsi_abmil"
    token_dir = attn_dir / f"seed_{seed}" / "exp1_2b_wsi_tsr_clinical_token"
    if not abmil_dir.exists() or not token_dir.exists():
        print("[Stage 4] Attention directories missing; skipping heatmap overlays.")
        return

    avail_slides = sorted(p.stem for p in token_dir.glob("*.npz") if (abmil_dir / f"{p.stem}.npz").exists())
    if not avail_slides:
        return

    sub_priors = priors_df[priors_df["slide_id"].isin(avail_slides)].sort_values("pTSR")
    if len(sub_priors) <= n_slides:
        chosen = sub_priors["slide_id"].tolist()
    else:
        idxs = np.linspace(0, len(sub_priors) - 1, num=n_slides, dtype=int)
        chosen = sub_priors.iloc[idxs]["slide_id"].tolist()

    bg_audit_rows: list[dict[str, Any]] = []
    for slide_id in chosen:
        d_abmil = np.load(abmil_dir / f"{slide_id}.npz")
        d_token = np.load(token_dir / f"{slide_id}.npz")
        thumb, dims0 = _load_thumb_rgb(slide_id, wsi_dir, thumbs_dir)
        if thumb is None or dims0 is None:
            continue

        th_h, th_w = thumb.shape[:2]
        w0, h0 = dims0
        scale_x = th_w / max(float(w0), 1.0)
        scale_y = th_h / max(float(h0), 1.0)

        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
        axes[0].imshow(thumb)
        axes[0].set_title(f"{slide_id}\nH&E (pTSR={float(d_token['pTSR'][0]):.3f})", fontsize=9.5)
        axes[0].axis("off")

        for ax, data, title in [
            (axes[1], d_abmil, "Exp 1.1: WSI-only ABMIL Attention"),
            (axes[2], d_token, "Exp 1.2b: TSR-Token Cross-Attention"),
        ]:
            coords = data["coords"]
            weights = data["attn_weights"].astype(np.float64)
            w_norm = (weights - weights.min()) / (weights.max() - weights.min() + 1e-12)

            heat = np.zeros((th_h, th_w), dtype=np.float32)
            counts = np.zeros((th_h, th_w), dtype=np.float32)
            tile_w = max(int(round(256 * scale_x)), 4)
            tile_h = max(int(round(256 * scale_y)), 4)

            bg_top_hits = 0
            top_k = max(1, int(np.ceil(0.10 * len(w_norm))))
            top_idx = set(np.argsort(w_norm)[-top_k:].tolist())

            for p_i, ((x0, y0), val) in enumerate(zip(coords, w_norm)):
                px = int(np.clip(round(x0 * scale_x), 0, th_w - 1))
                py = int(np.clip(round(y0 * scale_y), 0, th_h - 1))
                px2 = min(th_w, px + tile_w)
                py2 = min(th_h, py + tile_h)
                heat[py:py2, px:px2] += float(val)
                counts[py:py2, px:px2] += 1.0
                if p_i in top_idx:
                    patch_rgb = thumb[py:py2, px:px2]
                    if patch_rgb.size > 0 and float(patch_rgb.mean()) > bg_rgb_thresh:
                        bg_top_hits += 1

            bg_frac = bg_top_hits / float(top_k)
            bg_audit_rows.append(
                {"slide_id": slide_id, "model": title, "top10pct_bg_fraction": round(bg_frac, 4)}
            )
            if bg_frac > bg_warn_frac:
                print(
                    f"[RED-LINE WARNING] {slide_id} ({title}): {bg_frac:.1%} of top-10% attention "
                    "patches fall on bright glass background! Fix Stage 2 tissue segmentation."
                )

            heat_avg = np.divide(heat, np.maximum(counts, 1.0))
            ax.imshow(thumb)
            ax.imshow(np.ma.masked_where(counts == 0, heat_avg), cmap="jet", alpha=0.52, vmin=0.0, vmax=1.0)
            ax.set_title(f"{title}\n(BG overlap={bg_frac:.1%})", fontsize=9.5)
            ax.axis("off")

        fig.tight_layout()
        fig.savefig(heat_dir / f"{slide_id}_overlay.png", dpi=180)
        plt.close(fig)

    if bg_audit_rows:
        pd.DataFrame(bg_audit_rows).to_csv(report_dir / "heatmap_background_audit.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4: Evaluation statistics, RCS, SHAP, and WSI overlays")
    parser.add_argument("--config", type=str, default=str(DIR1_ROOT / "configs/tcga_lihc_demo.yaml"))
    parser.add_argument("--ablation-dir", type=str, default=None)
    parser.add_argument("--priors-csv", type=str, default=None)
    parser.add_argument("--attention-dir", type=str, default=None)
    parser.add_argument("--report-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    if args.synthetic:
        ablation_dir = DIR1_ROOT / "outputs/synthetic/ablation"
        priors_path = DIR1_ROOT / "outputs/synthetic/priors/slide_priors.csv"
        attn_dir = DIR1_ROOT / "outputs/synthetic/attention"
        report_dir = DIR1_ROOT / "outputs/synthetic/report"
        wsi_dir = None
        thumbs_dir = DIR1_ROOT / "data/synthetic/thumbnails"
    else:
        ablation_dir = Path(args.ablation_dir) if args.ablation_dir else resolve_path(DIR1_ROOT, cfg["paths"]["ablation_dir"])
        priors_path = Path(args.priors_csv) if args.priors_csv else resolve_path(DIR1_ROOT, cfg["paths"]["priors_csv"])
        attn_dir = Path(args.attention_dir) if args.attention_dir else resolve_path(DIR1_ROOT, cfg["paths"]["attention_dir"])
        report_dir = Path(args.report_dir) if args.report_dir else resolve_path(DIR1_ROOT, cfg["paths"]["report_dir"])
        wsi_dir = resolve_path(DIR1_ROOT, cfg["paths"]["wsi_dir"])
        thumbs_dir = resolve_path(DIR1_ROOT, cfg["paths"].get("thumbnails_dir", "data/thumbnails"))

    report_dir.mkdir(parents=True, exist_ok=True)
    fold_df = pd.read_csv(ablation_dir / "fold_metrics.csv")
    oof_df = pd.read_csv(ablation_dir / "oof_predictions.csv")
    priors_df = pd.read_csv(priors_path)

    clinical_cols = list(cfg["data"]["clinical_cols"])
    default_seed = int(args.seed or fold_df["seed"].iloc[0])
    target_exp = "exp1_2b_wsi_tsr_clinical_token"

    cindex_table = build_cindex_report(fold_df, oof_df, report_dir)
    plot_km_curves(oof_df, report_dir)
    mv_cox = run_multivariable_cox(oof_df, clinical_cols, target_exp, report_dir)

    eval_cfg = cfg.get("evaluation", {})
    plot_rcs_tsr_curve(
        oof_df=oof_df,
        knots_pct=list(eval_cfg.get("rcs_knots_percentiles", [5, 35, 65, 95])),
        report_dir=report_dir,
    )
    plot_shap_summary(oof_df, clinical_cols, target_exp, report_dir)
    render_attention_heatmaps(
        oof_df=oof_df,
        priors_df=priors_df,
        attn_dir=attn_dir,
        wsi_dir=wsi_dir,
        thumbs_dir=thumbs_dir,
        seed=default_seed,
        n_slides=int(eval_cfg.get("heatmap_num_slides", 4)),
        bg_rgb_thresh=int(eval_cfg.get("bg_rgb_threshold", 225)),
        bg_warn_frac=float(eval_cfg.get("bg_warn_fraction", 0.20)),
        report_dir=report_dir,
    )

    print(f"\n[Stage 4] Generated verification report under {report_dir}")
    print("\n--- Ablation C-index Summary ---")
    print(cindex_table.to_string(index=False))
    print("\n--- Multivariable Cox Summary ---")
    print(mv_cox.round(4).to_string())


if __name__ == "__main__":
    main()
