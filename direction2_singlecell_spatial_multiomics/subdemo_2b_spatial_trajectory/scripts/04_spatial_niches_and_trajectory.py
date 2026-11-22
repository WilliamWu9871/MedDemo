#!/usr/bin/env python3
"""Sub-demo 2B Step 4: Spatial Niche clustering, NMF Tissue Microenvironments, and DPT spatial trajectory.

Produces the core Paper 3 deliverables:
  1. Deconvolved cell-state abundances overlaid semi-transparently on H&E histology
  2. Spatial Niche (Leiden on [P, P_nbr]) & NMF Tissue Microenvironments (GBM_space_paper 2.3)
  3. Diffusion Pseudotime (scanpy.tl.dpt, or optional scFates) projected onto 2D Visium tissue coordinates
"""

from __future__ import annotations

import argparse
from pathlib import Path

SUB2B_ROOT = Path(__file__).resolve().parents[1]


def _pick_root_spot_index(adata_tumor, root_candidates: list[str], state_cols: list[str]) -> int:
    import numpy as np

    for cand in root_candidates:
        matches = [c for c in state_cols if cand.lower() in c.lower()]
        if matches:
            score = adata_tumor.obs[matches].sum(axis=1).to_numpy()
            return int(np.argmax(score))
    # Fallback: first spot
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Sub-demo 2B Step 4: Spatial Niches & Trajectory projection")
    parser.add_argument("--config", type=str, default=str(SUB2B_ROOT / "configs/gbm_spatial_demo.yaml"))
    parser.add_argument("--deconv-h5ad", type=str, default=None)
    parser.add_argument("--use-scfates", action="store_true", help="Optionally fit scFates principal tree (Paper 3 p.18)")
    args = parser.parse_args()

    import anndata as ad
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import squidpy as sq
    import yaml
    from sklearn.decomposition import NMF

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    c2l_dir = SUB2B_ROOT / cfg["paths"]["cell2location_dir"]
    fig_dir = SUB2B_ROOT / cfg["paths"]["figures_dir"]
    tab_dir = SUB2B_ROOT / cfg["paths"]["tables_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    in_h5ad = Path(args.deconv_h5ad) if args.deconv_h5ad else (c2l_dir / "visium_deconvolved.h5ad")
    adata = sc.read_h5ad(in_h5ad)

    abund_df = pd.DataFrame(adata.obsm["q05_cell_abundance_w_sf"], index=adata.obs_names)
    state_cols = abund_df.columns.tolist()
    P = abund_df.div(abund_df.sum(axis=1).replace(0.0, 1.0), axis=0).to_numpy(dtype=np.float32)

    # 1. Spatial Niche Partitioning via [Spot proportions, Spatial neighbor mean proportions] -> Leiden
    if "spatial_connectivities" not in adata.obsp:
        sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)
    A = adata.obsp["spatial_connectivities"].tocsr()
    row_sum = np.asarray(A.sum(axis=1)).ravel()
    row_sum[row_sum == 0] = 1.0
    A_norm = A.multiply(1.0 / row_sum[:, None])
    P_nbr = A_norm @ P

    niche_feat = np.concatenate([P, P_nbr], axis=1)
    adata_niche = ad.AnnData(X=niche_feat, obs=adata.obs.copy())
    sc.pp.neighbors(adata_niche, use_rep="X", n_neighbors=min(15, max(5, adata.n_obs // 20)))
    sc.tl.leiden(adata_niche, resolution=float(cfg["niches"]["leiden_resolution"]), key_added="spatial_niche")
    adata.obs["spatial_niche"] = adata_niche.obs["spatial_niche"].astype(str).map(lambda s: f"Niche_{s}")

    # 2. Paper 3 NMF Tissue Microenvironments (TMs, mirroring GBM_space_paper 2.3.cell2location_NMF)
    n_tm = min(int(cfg["niches"]["nmf_n_factors"]), len(state_cols), max(2, adata.n_obs // 10))
    nmf = NMF(n_components=n_tm, init="nndsvda", random_state=42, max_iter=500)
    tm_W = nmf.fit_transform(np.maximum(abund_df.to_numpy(dtype=np.float32), 0.0))
    adata.obsm["nmf_tm_weights"] = tm_W
    adata.obs["dominant_tm"] = pd.Categorical([f"TM_{k}" for k in np.argmax(tm_W, axis=1)])

    # 3. Diffusion Pseudotime Trajectory (scanpy.tl.dpt) on high-tumor-purity spots -> Spatial Projection
    traj_cfg = cfg["trajectory"]
    prefixes = tuple(p.lower() for p in traj_cfg.get("malignant_prefixes", []))
    malig_cols = [c for c in state_cols if c.lower().startswith(prefixes)]
    if malig_cols:
        tumor_purity = pd.DataFrame(P, index=adata.obs_names, columns=state_cols)[malig_cols].sum(axis=1)
        high_purity_mask = (tumor_purity >= float(traj_cfg.get("min_tumor_purity", 0.30))).to_numpy()
        if int(high_purity_mask.sum()) < 30:
            high_purity_mask = np.ones(adata.n_obs, dtype=bool)
    else:
        high_purity_mask = np.ones(adata.n_obs, dtype=bool)

    adata_traj = adata[high_purity_mask].copy()
    if "log1p_norm" in adata_traj.layers:
        adata_traj.X = adata_traj.layers["log1p_norm"].copy()
    else:
        sc.pp.normalize_total(adata_traj, target_sum=1e4)
        sc.pp.log1p(adata_traj)

    n_pcs = min(30, adata_traj.n_vars - 1, adata_traj.n_obs - 1)
    sc.pp.pca(adata_traj, n_comps=max(5, n_pcs))
    sc.pp.neighbors(adata_traj, n_neighbors=min(15, max(5, adata_traj.n_obs // 15)))
    sc.tl.diffmap(adata_traj)

    root_idx = _pick_root_spot_index(adata_traj, list(traj_cfg["root_state_candidates"]), state_cols)
    adata_traj.uns["iroot"] = root_idx
    sc.tl.dpt(adata_traj)

    if args.use_scfates:
        try:
            import scFates as scf

            scf.tl.tree(adata_traj, method="ppt", Nodes=min(30, max(5, adata_traj.n_obs // 20)), sigma=0.1, lam=10, seed=42)
            scf.tl.root(adata_traj, root=adata_traj.obs_names[root_idx])
            scf.tl.pseudotime(adata_traj)
            print("[2B Step 4] Fitted optional scFates principal tree.")
        except Exception as exc:
            print(f"[2B Step 4] scFates skipped ({exc}); using Scanpy DPT pseudotime.")

    adata.obs["dpt_pseudotime"] = np.nan
    adata.obs.loc[adata_traj.obs_names, "dpt_pseudotime"] = adata_traj.obs["dpt_pseudotime"].to_numpy()

    # Export summary tables
    comp_df = pd.DataFrame(P, index=adata.obs_names, columns=state_cols)
    comp_df["spatial_niche"] = adata.obs["spatial_niche"].to_numpy()
    niche_comp = comp_df.groupby("spatial_niche").mean().round(4)
    niche_comp.to_csv(tab_dir / "niche_composition.csv")

    pt_summary = (
        adata.obs.groupby("spatial_niche")["dpt_pseudotime"]
        .agg(["count", "mean", "median", "std"])
        .round(4)
    )
    pt_summary.to_csv(tab_dir / "pseudotime_summary.csv")

    # 4. H&E Overlay Visualizations
    plot_states = state_cols[: min(4, len(state_cols))]
    fig, axes = plt.subplots(1, len(plot_states), figsize=(4.2 * len(plot_states), 4.0))
    if len(plot_states) == 1:
        axes = [axes]
    for ax, st in zip(axes, plot_states):
        sc.pl.spatial(adata, color=st, ax=ax, show=False, alpha_img=0.65, cmap="magma", title=f"{st} (q05 abundance)")
    fig.tight_layout()
    fig.savefig(fig_dir / "he_overlay_cell_abundances.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    sc.pl.spatial(adata, color="spatial_niche", ax=axes[0], show=False, alpha_img=0.65, title="Spatial Niches (Leiden)")
    sc.pl.spatial(adata, color="dominant_tm", ax=axes[1], show=False, alpha_img=0.65, title="NMF Tissue Microenvironments (TMs)")
    fig.tight_layout()
    fig.savefig(fig_dir / "he_overlay_spatial_niches.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    sc.pl.spatial(
        adata,
        color="dpt_pseudotime",
        ax=axes[0],
        show=False,
        alpha_img=0.65,
        cmap="viridis",
        title="Spatial Trajectory Projection (DPT Pseudotime)",
    )
    valid_pt = adata.obs.dropna(subset=["dpt_pseudotime"])
    valid_pt.boxplot(column="dpt_pseudotime", by="spatial_niche", ax=axes[1], rot=25)
    axes[1].set_title("Pseudotime Across Spatial Niches")
    axes[1].set_ylabel("DPT Pseudotime")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(fig_dir / "spatial_pseudotime_trajectory.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    adata.write_h5ad(c2l_dir / "visium_niches_trajectory.h5ad")
    print(f"[2B Step 4] Complete! Outputs saved under {fig_dir} & {tab_dir}")
    print("\n--- Niche Pseudotime Summary ---")
    print(pt_summary.to_string())


if __name__ == "__main__":
    main()
