#!/usr/bin/env python3
"""Sub-demo 2B Step 2: Gene ID alignment check (Symbol vs Ensembl) + Squidpy spatial Moran's I."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

SUB2B_ROOT = Path(__file__).resolve().parents[1]


def _load_detect_gene_id_type():
    spec = importlib.util.spec_from_file_location(
        "step1_mod", SUB2B_ROOT / "scripts/01_download_spatial_and_ref.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.detect_gene_id_type


def _harmonize_to_symbols(adata, label: str):
    """Ensure var_names are uppercase-consistent gene symbols when symbol columns exist."""
    detect_fn = _load_detect_gene_id_type()
    id_type = detect_fn(adata.var_names[: min(200, adata.n_vars)])
    print(f"[2B Step 2] {label} var_names detected as '{id_type}' ({adata.n_vars} genes).")

    if id_type == "ensembl":
        for col in ["SYMBOL", "gene_symbols", "feature_name", "gene_name", "symbol"]:
            if col in adata.var.columns:
                symbols = adata.var[col].astype(str)
                if detect_fn(symbols[: min(200, len(symbols))]) == "symbol":
                    adata.var["ensembl_id"] = adata.var_names.astype(str)
                    adata.var_names = symbols
                    adata.var_names_make_unique()
                    print(f"[2B Step 2] Remapped {label} var_names from Ensembl -> column {col!r}.")
                    return adata
        # Strip Ensembl version suffix (e.g. ENSG00000146648.12 -> ENSG00000146648)
        adata.var_names = adata.var_names.astype(str).str.replace(r"\.\d+$", "", regex=True)
        adata.var_names_make_unique()
    return adata


def main() -> None:
    parser = argparse.ArgumentParser(description="Sub-demo 2B Step 2: Gene ID alignment & Squidpy Moran's I")
    parser.add_argument("--config", type=str, default=str(SUB2B_ROOT / "configs/gbm_spatial_demo.yaml"))
    parser.add_argument("--visium", type=str, default=None)
    parser.add_argument("--ref", type=str, default=None)
    parser.add_argument("--allow-small-panel", action="store_true", help="Allow <8000 shared genes (e.g., synthetic)")
    parser.add_argument("--synthetic", action="store_true", help="Alias for synthetic small-panel run")
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import scanpy as sc
    import squidpy as sq
    import yaml

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    proc_dir = SUB2B_ROOT / cfg["paths"]["processed_dir"]
    vis_path = Path(args.visium) if args.visium else (proc_dir / "visium_sample.h5ad")
    ref_path = Path(args.ref) if args.ref else (proc_dir / "reference_sc.h5ad")

    adata_vis = _harmonize_to_symbols(sc.read_h5ad(vis_path), "Visium")
    adata_vis.var_names_make_unique()

    if ref_path.exists():
        adata_ref = _harmonize_to_symbols(sc.read_h5ad(ref_path), "Reference scRNA-seq")
        adata_ref.var_names_make_unique()
        shared_genes = adata_vis.var_names.intersection(adata_ref.var_names)
        n_shared = len(shared_genes)
        min_shared = int(cfg["qc"]["min_shared_genes"])
        print(f"[2B Step 2] Shared genes (Visium ∩ scRNA-seq Reference): {n_shared}")

        if n_shared < min_shared and not (args.allow_small_panel or args.synthetic):
            raise RuntimeError(
                f"[RED-LINE ERROR] Gene overlap too small: {n_shared} < {min_shared}! "
                "Likely cause: Gene Symbol (e.g., EGFR) vs Ensembl ID (e.g., ENSG00000146648) mismatch "
                "between Visium and scRNA-seq reference. Check .var columns or pass --allow-small-panel if intentional."
            )
        adata_ref.write_h5ad(proc_dir / "reference_sc_aligned.h5ad")
    else:
        print(f"[2B Step 2] Warning: Reference {ref_path} not found yet; running Squidpy on Visium only.")

    # Preserve raw counts in .X while creating log1p_norm layer for Moran's I & spatial plots
    adata_norm = adata_vis.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)
    adata_vis.layers["log1p_norm"] = adata_norm.X.copy()

    sq.gr.spatial_neighbors(adata_vis, coord_type="generic", delaunay=True)

    # Calculate Moran's I across top variable + marker genes
    sc.pp.highly_variable_genes(adata_norm, n_top_genes=min(1000, adata_norm.n_vars), flavor="seurat")
    hvg_genes = adata_norm.var_names[adata_norm.var["highly_variable"]].tolist()
    markers = [g for g in cfg["qc"]["marker_genes"] if g in adata_vis.var_names]
    test_genes = list(dict.fromkeys(markers + hvg_genes[:200]))

    sq.gr.spatial_autocorr(adata_vis, mode="moran", genes=test_genes, layer="log1p_norm", n_perms=100)

    squid_dir = SUB2B_ROOT / cfg["paths"]["squidpy_dir"]
    fig_dir = SUB2B_ROOT / cfg["paths"]["figures_dir"]
    squid_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    moran_df = adata_vis.uns["moranI"].copy()
    moran_df.to_csv(squid_dir / "moran_i_scores.csv")
    adata_vis.write_h5ad(proc_dir / "visium_qc.h5ad")

    marker_table = moran_df.loc[[m for m in markers if m in moran_df.index]]
    if not marker_table.empty:
        print("\n[2B Step 2] Microenvironment Marker Moran's I:")
        print(marker_table[["I", "pval_norm"]].round(4).to_string())

    plot_genes = [g for g in markers[:4] if g in adata_vis.var_names]
    if plot_genes:
        fig, axes = plt.subplots(1, len(plot_genes), figsize=(4.2 * len(plot_genes), 4.0))
        if len(plot_genes) == 1:
            axes = [axes]
        for ax, gene in zip(axes, plot_genes):
            sc.pl.spatial(
                adata_vis,
                color=gene,
                layer="log1p_norm",
                ax=ax,
                show=False,
                alpha_img=0.65,
                title=f"{gene} (I={moran_df.loc[gene, 'I']:.2f})" if gene in moran_df.index else gene,
            )
        fig.tight_layout()
        fig.savefig(fig_dir / "squidpy_spatial_markers.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    print(f"[2B Step 2] Saved QC Visium -> {proc_dir / 'visium_qc.h5ad'} & Moran's I -> {squid_dir / 'moran_i_scores.csv'}")


if __name__ == "__main__":
    main()
