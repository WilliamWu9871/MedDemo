#!/usr/bin/env python3
"""Sub-demo 2A Step 2: Unconstrained multimodal VAE baseline using scvi-tools MultiVI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SUB2A_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sub-demo 2A Step 2: Run MultiVI baseline")
    parser.add_argument("--config", type=str, default=str(SUB2A_ROOT / "configs/pbmc_demo.yaml"))
    parser.add_argument("--mudata", type=str, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--synthetic", action="store_true", help="Fast smoke test epochs")
    args = parser.parse_args()

    import anndata as ad
    import matplotlib.pyplot as plt
    import muon as mu
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import scvi
    import yaml
    from sklearn.metrics import silhouette_score

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    proc_dir = SUB2A_ROOT / cfg["data"]["processed_dir"]
    mdata_path = Path(args.mudata) if args.mudata else (proc_dir / "multiome.h5mu")
    mdata = mu.read_h5mu(mdata_path)

    adata_rna = mdata.mod["rna"].copy()
    adata_atac = mdata.mod["atac"].copy()

    # Use HVGs for RNA when available to match standard MultiVI memory footprint
    if "highly_variable" in adata_rna.var and int(adata_rna.var["highly_variable"].sum()) > 100:
        adata_rna = adata_rna[:, adata_rna.var["highly_variable"]].copy()

    adata_rna.var["modality"] = "Gene Expression"
    adata_atac.var["modality"] = "Peaks"

    batch_col = cfg["data"].get("batch_col", "sample")
    celltype_col = cfg["data"].get("celltype_col", "celltype")

    adata_paired = ad.concat([adata_rna, adata_atac], axis=1, merge="first")
    adata_paired.obs[batch_col] = adata_rna.obs[batch_col].astype(str)
    if celltype_col in adata_rna.obs:
        adata_paired.obs[celltype_col] = adata_rna.obs[celltype_col].astype(str)
    adata_paired.var["modality"] = (
        ["Gene Expression"] * adata_rna.n_vars + ["Peaks"] * adata_atac.n_vars
    )
    adata_paired = scvi.data.organize_multiome_anndatas(adata_paired)

    mv_cfg = cfg["multivi"]
    out_dir = SUB2A_ROOT / mv_cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    max_epochs = int(args.max_epochs or (20 if args.synthetic else mv_cfg["max_epochs"]))
    n_latent = int(mv_cfg["n_latent"])
    batch_size = int(mv_cfg["batch_size"])
    leiden_res = float(mv_cfg["leiden_resolution"])

    scvi.settings.seed = 42
    scvi.model.MULTIVI.setup_anndata(adata_paired, batch_key=batch_col)
    model = scvi.model.MULTIVI(
        adata_paired,
        n_genes=adata_rna.n_vars,
        n_regions=adata_atac.n_vars,
        n_latent=n_latent,
    )
    model.train(max_epochs=max_epochs, batch_size=batch_size)

    latent = model.get_latent_representation()
    adata_out = adata_rna.copy()
    adata_out.obsm["X_multivi"] = latent
    sc.pp.neighbors(adata_out, use_rep="X_multivi")
    sc.tl.leiden(adata_out, resolution=leiden_res, key_added="leiden_multivi")
    sc.tl.umap(adata_out)

    adata_out.write_h5ad(out_dir / "multivi_latent.h5ad")

    color_cols = ["leiden_multivi"]
    if celltype_col in adata_out.obs and adata_out.obs[celltype_col].nunique() > 1:
        color_cols.append(celltype_col)

    fig = sc.pl.umap(adata_out, color=color_cols, show=False, return_fig=True)
    fig.savefig(out_dir / "multivi_umap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    sil_leiden = float(silhouette_score(latent, adata_out.obs["leiden_multivi"]))
    metrics = {
        "n_cells": int(adata_out.n_obs),
        "n_genes": int(adata_rna.n_vars),
        "n_peaks": int(adata_atac.n_vars),
        "n_latent": n_latent,
        "n_leiden_clusters": int(adata_out.obs["leiden_multivi"].nunique()),
        "silhouette_leiden": round(sil_leiden, 4),
    }
    if celltype_col in adata_out.obs and adata_out.obs[celltype_col].nunique() > 1:
        metrics["silhouette_celltype"] = round(
            float(silhouette_score(latent, adata_out.obs[celltype_col])), 4
        )

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[2A Step 2] MultiVI baseline complete -> {out_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
