#!/usr/bin/env python3
"""Sub-demo 2A Step 1: Dual-modality QC, barcode intersection, and MuData + raw h5ad export.

Keeps raw integer counts in `.X` (required by MultiVI and bioFAM/scDoRI) while storing
log1p-normalized RNA in `layers['log1p_norm']` and TF-IDF ATAC in `layers['tfidf']`.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

SUB2A_ROOT = Path(__file__).resolve().parents[1]

TENX_PBMC_URLS = {
    "pbmc3k": "https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_3k/pbmc_granulocyte_sorted_3k_filtered_feature_bc_matrix.h5",
    "pbmc10k": "https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_unsorted_10k/pbmc_unsorted_10k_filtered_feature_bc_matrix.h5",
}

_PEAK_SEP_RE = re.compile(r"^(chr[0-9XYM]+)[-_:](\d+)[-_:](\d+)$", re.IGNORECASE)


def normalize_peak_name(peak: str) -> str:
    """Normalize ATAC peak names to standard ``chr:start-end`` format required by scDoRI."""
    if not isinstance(peak, str):
        raise TypeError(f"Expected string peak identifier, got {type(peak).__name__}")
    s = peak.strip()
    m = _PEAK_SEP_RE.match(s)
    if m:
        chrom, start, end = m.group(1), m.group(2), m.group(3)
        return f"{chrom}:{start}-{end}"
    return s


def _download_file(url: str, dest: Path) -> Path:
    import requests
    from tqdm import tqdm

    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[2A Step 1] Downloading {url} -> {dest}")
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        tmp = dest.with_suffix(".part")
        with tmp.open("wb") as handle, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    pbar.update(len(chunk))
        tmp.rename(dest)
    return dest


def _build_synthetic_multiome(seed: int = 42):
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    n_per_state = 300
    states = ["T_cell", "B_cell", "Myeloid", "Progenitor_GBM"]
    state_tfs = {
        "T_cell": ["TCF7", "LEF1", "GATA3"],
        "B_cell": ["PAX5", "EBF1", "BCL11A"],
        "Myeloid": ["CEBPB", "SPI1", "IRF8"],
        "Progenitor_GBM": ["MYT1L", "SOX2", "OLIG2"],
    }
    all_tfs = [tf for tfs in state_tfs.values() for tf in tfs]
    other_genes = [f"GENE{i:04d}" for i in range(1, 289)]
    gene_names = all_tfs + other_genes  # 300 genes
    n_genes = len(gene_names)

    n_peaks = 600
    peak_names = [f"chr{(i % 22) + 1}:{(i + 1) * 2000}-{(i + 1) * 2000 + 500}" for i in range(n_peaks)]

    n_cells = n_per_state * len(states)
    rna_counts = rng.poisson(0.4, size=(n_cells, n_genes)).astype(np.float32)
    atac_counts = rng.binomial(1, 0.04, size=(n_cells, n_peaks)).astype(np.float32)

    cell_types = []
    barcodes = []
    for s_idx, st in enumerate(states):
        start_c = s_idx * n_per_state
        end_c = (s_idx + 1) * n_per_state
        # Block of co-regulated genes + state TFs
        g_start = 12 + s_idx * 45
        g_end = g_start + 45
        for tf in state_tfs[st]:
            tf_idx = gene_names.index(tf)
            rna_counts[start_c:end_c, tf_idx] += rng.poisson(6.0, size=n_per_state)
        rna_counts[start_c:end_c, g_start:g_end] += rng.poisson(4.5, size=(n_per_state, 45))

        p_start = s_idx * 110
        p_end = p_start + 110
        atac_counts[start_c:end_c, p_start:p_end] += rng.poisson(1.8, size=(n_per_state, 110))

        for k in range(n_per_state):
            cell_types.append(st)
            barcodes.append(f"CELL_{s_idx}_{k:04d}-1")

    obs = pd.DataFrame(
        {
            "celltype": pd.Categorical(cell_types),
            "sample": pd.Categorical(["syn_sample_1" if i % 2 == 0 else "syn_sample_2" for i in range(n_cells)]),
        },
        index=barcodes,
    )
    adata_rna = ad.AnnData(
        X=sp.csr_matrix(rna_counts),
        obs=obs.copy(),
        var=pd.DataFrame(index=gene_names),
    )
    adata_atac = ad.AnnData(
        X=sp.csr_matrix(atac_counts),
        obs=obs.copy(),
        var=pd.DataFrame(index=peak_names),
    )
    return adata_rna, adata_atac


def _load_10x_multiome(h5_path: Path, sample_name: str):
    import scanpy as sc

    adata = sc.read_10x_h5(h5_path, gex_only=False)
    adata.var_names_make_unique()
    ft = adata.var["feature_types"].astype(str)
    adata_rna = adata[:, ft == "Gene Expression"].copy()
    adata_atac = adata[:, ft == "Peaks"].copy()
    adata_rna.obs["sample"] = sample_name
    adata_atac.obs["sample"] = sample_name
    adata_rna.obs["celltype"] = "unannotated"
    adata_atac.obs["celltype"] = "unannotated"
    return adata_rna, adata_atac


def _subsample_stratified(adata_rna, adata_atac, max_cells: int, celltype_col: str, seed: int = 42):
    import numpy as np

    if adata_rna.n_obs <= max_cells:
        return adata_rna, adata_atac
    rng = np.random.default_rng(seed)
    if celltype_col in adata_rna.obs:
        chosen = []
        frac = max_cells / float(adata_rna.n_obs)
        for _, grp in adata_rna.obs.groupby(celltype_col, observed=True):
            n_take = max(1, int(round(len(grp) * frac)))
            chosen.extend(rng.choice(grp.index.to_numpy(), size=min(len(grp), n_take), replace=False).tolist())
        chosen = chosen[:max_cells]
    else:
        chosen = rng.choice(adata_rna.obs_names.to_numpy(), size=max_cells, replace=False).tolist()
    return adata_rna[chosen].copy(), adata_atac[chosen].copy()


def _compute_tfidf(X):
    import numpy as np
    import scipy.sparse as sp

    X_csr = X.tocsr().astype(np.float32) if sp.issparse(X) else sp.csr_matrix(X, dtype=np.float32)
    cell_sums = np.asarray(X_csr.sum(axis=1)).ravel()
    cell_sums[cell_sums == 0] = 1.0
    tf = X_csr.multiply(1.0 / cell_sums[:, None])

    n_cells = float(X_csr.shape[0])
    df = np.asarray((X_csr > 0).sum(axis=0)).ravel()
    df[df == 0] = 1.0
    idf = np.log1p(n_cells / df)
    tfidf = tf.multiply(idf[None, :])
    return tfidf.tocsr()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sub-demo 2A Step 1: Prepare paired RNA+ATAC MuData")
    parser.add_argument("--config", type=str, default=str(SUB2A_ROOT / "configs/pbmc_demo.yaml"))
    parser.add_argument("--source", type=str, default=None, choices=["pbmc3k", "pbmc10k", "h5ad", "synthetic"])
    parser.add_argument("--rna", type=str, default=None, help="Path to RNA .h5ad when --source h5ad")
    parser.add_argument("--atac", type=str, default=None, help="Path to ATAC .h5ad when --source h5ad")
    parser.add_argument("--subsample-cells", type=int, default=None, help="Optional max cells to retain")
    parser.add_argument("--cell-type-col", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import muon as mu
    import numpy as np
    import scanpy as sc
    import scipy.sparse as sp
    import yaml

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    source = args.source or cfg["data"]["source"]
    raw_dir = SUB2A_ROOT / cfg["data"]["raw_dir"]
    proc_dir = SUB2A_ROOT / cfg["data"]["processed_dir"]
    proc_dir.mkdir(parents=True, exist_ok=True)
    celltype_col = args.cell_type_col or cfg["data"].get("celltype_col", "celltype")
    batch_col = cfg["data"].get("batch_col", "sample")

    if source == "synthetic":
        adata_rna, adata_atac = _build_synthetic_multiome(seed=args.seed)
    elif source in TENX_PBMC_URLS:
        h5_path = _download_file(TENX_PBMC_URLS[source], raw_dir / f"{source}_filtered_feature_bc_matrix.h5")
        adata_rna, adata_atac = _load_10x_multiome(h5_path, sample_name=source)
    elif source == "h5ad":
        if not args.rna or not args.atac:
            raise ValueError("--rna and --atac are required when --source h5ad")
        adata_rna = sc.read_h5ad(args.rna)
        adata_atac = sc.read_h5ad(args.atac)
    else:
        raise ValueError(f"Unsupported source: {source}")

    # Intersect barcodes upfront before QC and ensure required obs columns
    common_init = adata_rna.obs_names.intersection(adata_atac.obs_names)
    adata_rna = adata_rna[common_init].copy()
    adata_atac = adata_atac[common_init].copy()

    if batch_col not in adata_rna.obs:
        adata_rna.obs[batch_col] = cfg["data"].get("sample_name", source)
    if celltype_col not in adata_rna.obs:
        adata_rna.obs[celltype_col] = "unannotated"
    adata_atac.obs[batch_col] = adata_rna.obs[batch_col].astype(str)
    adata_atac.obs[celltype_col] = adata_rna.obs[celltype_col].astype(str)

    if args.subsample_cells:
        adata_rna, adata_atac = _subsample_stratified(
            adata_rna, adata_atac, max_cells=args.subsample_cells, celltype_col=celltype_col, seed=args.seed
        )

    # Preserve raw integer counts strictly in .X
    if sp.issparse(adata_rna.X):
        adata_rna.X = adata_rna.X.tocsr()
    else:
        adata_rna.X = sp.csr_matrix(adata_rna.X)
    if sp.issparse(adata_atac.X):
        adata_atac.X = adata_atac.X.tocsr()
    else:
        adata_atac.X = sp.csr_matrix(adata_atac.X)

    qc = cfg["qc"]
    if source == "synthetic":
        min_genes = 20
        min_counts = 30
        n_top_genes = min(250, adata_rna.n_vars)
        min_cells_peak = 3
    else:
        min_genes = int(qc["min_genes_rna"])
        min_counts = int(qc["min_counts_rna"])
        n_top_genes = int(qc["n_top_genes"])
        min_cells_peak = int(qc["min_cells_peak"])

    # RNA QC
    adata_rna.var["mt"] = adata_rna.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata_rna, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    rna_mask = (
        (adata_rna.obs["n_genes_by_counts"] >= min_genes)
        & (adata_rna.obs["n_genes_by_counts"] <= int(qc["max_genes_rna"]))
        & (adata_rna.obs["total_counts"] >= min_counts)
        & (adata_rna.obs["pct_counts_mt"] <= float(qc["max_pct_mt"]))
    )
    adata_rna = adata_rna[rna_mask].copy()
    sc.pp.filter_genes(adata_rna, min_cells=3)

    # Normalized RNA goes into layer; raw counts remain in .X
    tmp_rna = adata_rna.copy()
    sc.pp.normalize_total(tmp_rna, target_sum=1e4)
    sc.pp.log1p(tmp_rna)
    adata_rna.layers["log1p_norm"] = tmp_rna.X.copy()
    sc.pp.highly_variable_genes(tmp_rna, n_top_genes=min(n_top_genes, tmp_rna.n_vars), flavor="seurat")
    adata_rna.var["highly_variable"] = tmp_rna.var["highly_variable"].to_numpy()

    # ATAC QC + chr:start-end normalization + TF-IDF layer
    adata_atac.var_names = [normalize_peak_name(p) for p in adata_atac.var_names]
    adata_atac.var_names_make_unique()
    sc.pp.filter_cells(adata_atac, min_genes=10 if source == "synthetic" else 100)
    sc.pp.filter_genes(adata_atac, min_cells=min_cells_peak)
    max_cells_peak = int(float(qc.get("max_cells_peak_frac", 0.90)) * adata_atac.n_obs)
    if max_cells_peak > min_cells_peak:
        n_cells_det = np.asarray((adata_atac.X > 0).sum(axis=0)).ravel()
        adata_atac = adata_atac[:, n_cells_det <= max_cells_peak].copy()

    # Final barcode intersection
    common = adata_rna.obs_names.intersection(adata_atac.obs_names)
    adata_rna = adata_rna[common].copy()
    adata_atac = adata_atac[common].copy()
    adata_atac.layers["tfidf"] = _compute_tfidf(adata_atac.X)

    # Ensure multi-group celltype annotations exist so scDoRI downstream clustermaps & silhouette never fail on 1 label
    if adata_rna.obs[celltype_col].astype(str).nunique() <= 1 and adata_rna.n_obs >= 20:
        tmp_cl = adata_rna.copy()
        tmp_cl.X = tmp_cl.layers["log1p_norm"].copy()
        if "highly_variable" in tmp_cl.var and int(tmp_cl.var["highly_variable"].sum()) >= 50:
            tmp_cl = tmp_cl[:, tmp_cl.var["highly_variable"]].copy()
        n_pcs = min(30, tmp_cl.n_vars - 1, tmp_cl.n_obs - 1)
        if n_pcs >= 2:
            sc.pp.pca(tmp_cl, n_comps=n_pcs)
            sc.pp.neighbors(tmp_cl, n_neighbors=min(15, max(5, tmp_cl.n_obs - 1)))
        else:
            sc.pp.neighbors(tmp_cl, use_rep="X", n_neighbors=min(15, max(5, tmp_cl.n_obs - 1)))
        sc.tl.leiden(tmp_cl, resolution=0.5, key_added="leiden_init")
        inferred = tmp_cl.obs["leiden_init"].astype(str).map(lambda s: f"Cluster_{s}")
        adata_rna.obs[celltype_col] = inferred.to_numpy()
        adata_atac.obs[celltype_col] = inferred.to_numpy()
        print(
            f"[2A Step 1] Inferred {adata_rna.obs[celltype_col].nunique()} Leiden clusters "
            f"into obs['{celltype_col}'] for unannotated cells."
        )

    # Export MuData + raw h5ads for scDoRI
    mdata = mu.MuData({"rna": adata_rna, "atac": adata_atac})
    mdata.write_h5mu(proc_dir / "multiome.h5mu")
    adata_rna.write_h5ad(proc_dir / "rna_raw.h5ad")
    adata_atac.write_h5ad(proc_dir / "atac_raw.h5ad")

    print(
        f"[2A Step 1] Prepared paired multiome ({source}): "
        f"{len(common)} cells | RNA={adata_rna.n_vars} genes ({int(adata_rna.var['highly_variable'].sum())} HVGs) "
        f"| ATAC={adata_atac.n_vars} peaks"
    )
    print(f"  Saved: {proc_dir / 'multiome.h5mu'}")
    print(f"  Saved: {proc_dir / 'rna_raw.h5ad'}, {proc_dir / 'atac_raw.h5ad'}")


if __name__ == "__main__":
    main()
