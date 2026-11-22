#!/usr/bin/env python3
"""Sub-demo 2B Step 3: Cell-state spatial deconvolution via BayraktarLab/cell2location.

Mirrors BayraktarLab/GBM_space_paper (2.1.run_cell2location.ipynb):
  Step A: Fit negative binomial RegressionModel on scRNA-seq reference to learn inf_aver signatures.
  Step B: Fit Cell2location spatial model on Visium counts (N_cells_per_location=30, detection_alpha=200).
  Audit : Run Moran's I on deconvolved cell abundances to catch salt-and-pepper noise red lines.
"""

from __future__ import annotations

import argparse
from pathlib import Path

SUB2B_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sub-demo 2B Step 3: Run cell2location deconvolution")
    parser.add_argument("--config", type=str, default=str(SUB2B_ROOT / "configs/gbm_spatial_demo.yaml"))
    parser.add_argument("--visium", type=str, default=None)
    parser.add_argument("--ref", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true", help="Use reduced epochs for quick smoke test")
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import squidpy as sq
    import torch
    import yaml
    from cell2location.models import Cell2location, RegressionModel
    from cell2location.utils.filtering import filter_genes

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    proc_dir = SUB2B_ROOT / cfg["paths"]["processed_dir"]
    c2l_dir = SUB2B_ROOT / cfg["paths"]["cell2location_dir"]
    fig_dir = SUB2B_ROOT / cfg["paths"]["figures_dir"]
    c2l_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    vis_path = Path(args.visium) if args.visium else (proc_dir / "visium_qc.h5ad")
    if not vis_path.exists():
        vis_path = proc_dir / "visium_sample.h5ad"
    ref_path = Path(args.ref) if args.ref else (proc_dir / "reference_sc_aligned.h5ad")
    if not ref_path.exists():
        ref_path = proc_dir / "reference_sc.h5ad"

    adata_vis = sc.read_h5ad(vis_path)
    adata_ref = sc.read_h5ad(ref_path)

    # Remove mitochondrial genes prior to cell2location (standard best practice)
    adata_vis = adata_vis[:, ~adata_vis.var_names.str.upper().str.startswith("MT-")].copy()
    adata_ref = adata_ref[:, ~adata_ref.var_names.str.upper().str.startswith("MT-")].copy()

    c2l_cfg = cfg["cell2location"]
    ref_cfg = cfg["reference"]
    cell_state_col = ref_cfg["cell_state_col"]
    ref_batch_col = ref_cfg["batch_col"]

    # Step A: Filter genes & fit RegressionModel on scRNA-seq reference
    if args.synthetic:
        selected = adata_ref.var_names.tolist()
    else:
        selected = filter_genes(
            adata_ref, cell_count_cutoff=5, cell_percentage_cutoff2=0.03, nonz_mean_cutoff=1.12
        )
        plt.close("all")
    adata_ref = adata_ref[:, selected].copy()

    use_gpu = torch.cuda.is_available()
    accelerator = "gpu" if use_gpu else "cpu"

    RegressionModel.setup_anndata(
        adata=adata_ref,
        batch_key=ref_batch_col if ref_batch_col in adata_ref.obs else None,
        labels_key=cell_state_col,
    )
    mod_ref = RegressionModel(adata_ref)
    ref_epochs = 40 if args.synthetic else int(c2l_cfg["ref_max_epochs"])
    mod_ref.train(
        max_epochs=ref_epochs,
        batch_size=min(int(c2l_cfg["ref_batch_size"]), adata_ref.n_obs),
        lr=float(c2l_cfg["ref_lr"]),
        accelerator=accelerator,
    )
    adata_ref = mod_ref.export_posterior(
        adata_ref,
        sample_kwargs={"num_samples": 200 if args.synthetic else 1000, "batch_size": 2500},
    )
    mod_ref.plot_history(0)
    plt.savefig(fig_dir / "cell2location_ref_elbo.png", dpi=160, bbox_inches="tight")
    plt.close()

    if "means_per_cluster_mu_fg" in adata_ref.varm:
        inf_aver = adata_ref.varm["means_per_cluster_mu_fg"][
            [f"means_per_cluster_mu_fg_{i}" for i in adata_ref.uns["mod"]["factor_names"]]
        ].copy()
    else:
        inf_aver = adata_ref.var[[f"means_per_cluster_mu_fg_{i}" for i in adata_ref.uns["mod"]["factor_names"]]].copy()
    inf_aver.columns = adata_ref.uns["mod"]["factor_names"]
    inf_aver.to_csv(c2l_dir / "reference_signatures_inf_aver.csv")

    # Step B: Intersect genes & fit Cell2location spatial model
    intersect = np.intersect1d(adata_vis.var_names, inf_aver.index)
    print(f"[2B Step 3] Fitting Cell2location on {adata_vis.n_obs} spots × {len(intersect)} shared genes...")
    adata_vis_sub = adata_vis[:, intersect].copy()
    inf_aver = inf_aver.loc[intersect].copy()

    Cell2location.setup_anndata(
        adata=adata_vis_sub,
        batch_key="sample" if "sample" in adata_vis_sub.obs else None,
    )
    mod_vis = Cell2location(
        adata_vis_sub,
        cell_state_df=inf_aver,
        N_cells_per_location=float(c2l_cfg["N_cells_per_location"]),
        detection_alpha=float(c2l_cfg["detection_alpha"]),
    )
    sp_epochs = 150 if args.synthetic else int(c2l_cfg["spatial_max_epochs"])
    mod_vis.train(
        max_epochs=sp_epochs,
        batch_size=min(int(c2l_cfg["spatial_batch_size"]), adata_vis_sub.n_obs),
        train_size=1.0,
        accelerator=accelerator,
    )
    adata_vis_sub = mod_vis.export_posterior(
        adata_vis_sub,
        sample_kwargs={"num_samples": 200 if args.synthetic else 1000, "batch_size": adata_vis_sub.n_obs},
    )
    mod_vis.plot_history(0)
    plt.savefig(fig_dir / "cell2location_spatial_elbo.png", dpi=160, bbox_inches="tight")
    plt.close()

    # Copy deconvolved posteriors onto full QC AnnData (retaining all genes + layers + spatial neighbors)
    abund_df = adata_vis_sub.obsm["q05_cell_abundance_w_sf"].copy()
    abund_df.columns = [c.replace("q05cell_abundance_w_sf_", "") for c in abund_df.columns]
    adata_vis.obsm["q05_cell_abundance_w_sf"] = abund_df
    for col in abund_df.columns:
        adata_vis.obs[col] = abund_df[col].to_numpy()

    # Salt-and-pepper spatial coherence check on deconvolved abundances
    if "spatial_connectivities" not in adata_vis.obsp:
        sq.gr.spatial_neighbors(adata_vis, coord_type="generic", delaunay=True)

    import anndata as ad

    abund_ad = ad.AnnData(X=abund_df.to_numpy(dtype=np.float32), obs=adata_vis.obs.copy(), var=pd.DataFrame(index=abund_df.columns))
    abund_ad.obsp["spatial_connectivities"] = adata_vis.obsp["spatial_connectivities"]
    abund_ad.obsp["spatial_distances"] = adata_vis.obsp["spatial_distances"]
    sq.gr.spatial_autocorr(abund_ad, mode="moran", genes=list(abund_df.columns), n_perms=50)
    med_moran = float(abund_ad.uns["moranI"]["I"].median())
    print(f"[2B Step 3] Deconvolved abundance median Moran's I = {med_moran:.3f}")
    if med_moran < float(c2l_cfg.get("min_moran_i_warn", 0.15)):
        print(
            f"[RED-LINE WARNING] Low spatial autocorrelation (median Moran's I={med_moran:.3f}) — "
            "check for salt-and-pepper noise (tune N_cells_per_location or set detection_alpha=20)."
        )

    out_h5ad = c2l_dir / "visium_deconvolved.h5ad"
    adata_vis.write_h5ad(out_h5ad)
    abund_df.to_csv(c2l_dir / "spot_cell_abundances_q05.csv")
    print(f"[2B Step 3] Saved deconvolved Visium -> {out_h5ad}")


if __name__ == "__main__":
    main()
