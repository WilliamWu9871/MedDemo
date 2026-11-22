#!/usr/bin/env python3
"""Sub-demo 2A Step 4: Sanity checks comparing MultiVI vs scDoRI & auditing eGRN quality.

Validates the 3 core acceptance checks:
  1. Representation separation: Side-by-side UMAP (MultiVI vs scDoRI), Silhouette & ARI
  2. Topic activity & collapse red-line audit: Active topics (max > 0.1), entropy, top-2 variance share
  3. Regulatory self-consistency: Top-2 Topics -> Top TFs + Top target genes DotPlot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

SUB2A_ROOT = Path(__file__).resolve().parents[1]


def _resolve_scdori_path(scdori_config_path: Path, raw_path: str) -> Path:
    """Resolve a path from a scDoRI JSON config the same way scdori.configuration does."""
    p = Path(raw_path)
    return p if p.is_absolute() else (scdori_config_path.parent / p).resolve()


def _topic_matches_value(val: Any, tid: int) -> bool:
    """Return True if a raw cell value matches topic ID ``tid`` ('Topic_{tid}' or '{tid}')."""
    s = str(val).strip()
    return s == f"Topic_{tid}" or s == str(tid)


def extract_topic_tfs(leading_tfs: Any, tid: int, top_n: int) -> list[str]:
    """Extract top TF symbols for topic ``tid`` from a DataFrame or sequence of dicts."""
    if hasattr(leading_tfs, "columns"):
        topic_col = "topic" if "topic" in leading_tfs.columns else leading_tfs.columns[0]
        tf_col = "tf" if "tf" in leading_tfs.columns else leading_tfs.columns[1]
        mask = [
            _topic_matches_value(v, tid)
            for v in leading_tfs[topic_col].tolist()
        ]
        sub = leading_tfs.loc[mask, tf_col].tolist()
        return [str(t).strip() for t in sub if str(t).strip()][:top_n]

    out: list[str] = []
    for row in leading_tfs:
        topic_val = row.get("topic") if "topic" in row else next(iter(row.values()))
        if _topic_matches_value(topic_val, tid):
            tf_val = str(row.get("tf", "")).strip()
            if tf_val:
                out.append(tf_val)
                if len(out) >= top_n:
                    break
    return out


def extract_topic_genes(leading_genes: Any, tid: int, top_n: int) -> list[str]:
    """Extract top target gene symbols for topic ``tid`` from DataFrame or dict rows.

    Handles both official scDoRI ``downstream.ipynb`` exports (``Topic`` + comma-separated
    ``Leading genes``) and one-row-per-gene tables (``topic`` + ``gene``).
    """
    if hasattr(leading_genes, "columns"):
        cols = list(leading_genes.columns)
        if "Topic" in cols and "Leading genes" in cols:
            genes: list[str] = []
            for topic_val, cell in zip(
                leading_genes["Topic"].tolist(),
                leading_genes["Leading genes"].tolist(),
            ):
                if _topic_matches_value(topic_val, tid) and cell is not None:
                    for g in str(cell).split(","):
                        g_clean = g.strip()
                        if g_clean and g_clean.lower() != "nan" and g_clean not in genes:
                            genes.append(g_clean)
                            if len(genes) >= top_n:
                                return genes
            return genes

        topic_col = "topic" if "topic" in cols else cols[0]
        gene_col = "gene" if "gene" in cols else cols[1]
        out: list[str] = []
        for topic_val, gene_val in zip(
            leading_genes[topic_col].tolist(),
            leading_genes[gene_col].tolist(),
        ):
            if _topic_matches_value(topic_val, tid):
                g_clean = str(gene_val).strip()
                if g_clean and g_clean.lower() != "nan":
                    out.append(g_clean)
                    if len(out) >= top_n:
                        break
        return out

    genes = []
    for row in leading_genes:
        if "Topic" in row and "Leading genes" in row:
            if _topic_matches_value(row["Topic"], tid) and row["Leading genes"] is not None:
                for g in str(row["Leading genes"]).split(","):
                    g_clean = g.strip()
                    if g_clean and g_clean not in genes:
                        genes.append(g_clean)
                        if len(genes) >= top_n:
                            return genes
        else:
            topic_val = row.get("topic") if "topic" in row else next(iter(row.values()))
            if _topic_matches_value(topic_val, tid):
                g_clean = str(row.get("gene", "")).strip()
                if g_clean:
                    genes.append(g_clean)
                    if len(genes) >= top_n:
                        break
    return genes


def _extract_scdori_topics_and_grn(
    scdori_config_path: Path,
    adata_ref,
) -> tuple[np.ndarray, list[str] | None, pd.DataFrame, pd.DataFrame]:
    """Load cell x Topic matrix, obs names, and leading TF/target tables from scDoRI outputs/model."""
    import numpy as np
    import pandas as pd

    with scdori_config_path.open("r", encoding="utf-8") as handle:
        sc_cfg = json.load(handle)

    grn_dir = _resolve_scdori_path(scdori_config_path, sc_cfg["downstream"]["grn_output"])
    topic_mat: np.ndarray | None = None
    topic_obs_names: list[str] | None = None

    for cand in [
        grn_dir / "matrix_exports" / "cell_topics.csv.gz",
        grn_dir / "matrix_exports" / "cell_topics.csv",
        grn_dir / "cell_topics.npy",
        grn_dir / "scdori_latent.npy",
    ]:
        if cand.exists():
            if cand.suffix == ".npy":
                topic_mat = np.load(cand)
            else:
                df = pd.read_csv(cand, index_col=0)
                topic_obs_names = df.index.astype(str).tolist()
                topic_mat = df.to_numpy(dtype=np.float32)
            break

    if topic_mat is None:
        try:
            import torch
            from torch.utils.data import DataLoader, TensorDataset
            from scdori import get_latent_topics, load_downstream_model
            from scdori.configuration import load_notebook_config

            settings = load_notebook_config(str(scdori_config_path))
            fit = load_downstream_model(settings)
            all_dataset = TensorDataset(torch.arange(fit.rna.n_obs))
            all_loader = DataLoader(
                all_dataset,
                batch_size=int(getattr(fit.config, "batch_size_cell_prediction", 256)),
                shuffle=False,
            )
            topic_mat = np.asarray(
                get_latent_topics(
                    fit.model,
                    fit.device,
                    all_loader,
                    fit.rna,
                    fit.atac,
                    fit.num_cells,
                    fit.tf_indices,
                    fit.onehot_batch,
                ),
                dtype=np.float32,
            )
            topic_obs_names = fit.rna.obs_names.astype(str).tolist()
        except Exception as exc:
            raise FileNotFoundError(
                f"Could not find or extract scDoRI cell×Topic matrix under {grn_dir} "
                f"(run scripts/03_run_scdori.sh first, or pass --synthetic to test diagnostics). Details: {exc}"
            ) from exc

    genes_path = grn_dir / "leading_topic_genes.csv"
    tfs_path = grn_dir / "tf_score_views" / "top_topic_tf_activator.csv"
    if not genes_path.exists() or not tfs_path.exists():
        raise FileNotFoundError(
            f"Missing scDoRI downstream tables ({genes_path} or {tfs_path}). "
            "Ensure downstream.ipynb completed cleanly."
        )
    leading_genes = pd.read_csv(genes_path)
    leading_tfs = pd.read_csv(tfs_path)
    return topic_mat, topic_obs_names, leading_tfs, leading_genes


def _generate_synthetic_scdori_outputs(adata_rna, n_topics: int = 16, seed: int = 42):
    """Synthesize non-collapsed scDoRI-compatible outputs aligned with synthetic multiome states."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    n_cells = adata_rna.n_obs
    states = adata_rna.obs["celltype"].astype(str).to_numpy()
    unique_states = sorted(np.unique(states).tolist())

    topic_mat = rng.dirichlet(np.full(n_topics, 0.18), size=n_cells).astype(np.float32)
    for idx, st in enumerate(unique_states):
        mask = states == st
        # Each state activates 2 primary regulatory topics strongly
        t1, t2 = (idx * 2) % n_topics, (idx * 2 + 1) % n_topics
        topic_mat[mask, t1] += rng.uniform(0.35, 0.65, size=mask.sum())
        topic_mat[mask, t2] += rng.uniform(0.20, 0.45, size=mask.sum())
    topic_mat /= topic_mat.sum(axis=1, keepdims=True)

    state_tfs = {
        "T_cell": ["TCF7", "LEF1", "GATA3"],
        "B_cell": ["PAX5", "EBF1", "BCL11A"],
        "Myeloid": ["CEBPB", "SPI1", "IRF8"],
        "Progenitor_GBM": ["MYT1L", "SOX2", "OLIG2"],
    }
    var_names = adata_rna.var_names.tolist()
    tf_rows, gene_rows = [], []
    for k in range(n_topics):
        st = unique_states[(k // 2) % len(unique_states)]
        tfs = [t for t in state_tfs.get(st, var_names[:3]) if t in var_names]
        if not tfs:
            tfs = var_names[:3]
        for rank, tf in enumerate(tfs):
            tf_rows.append({"topic": k, "tf": tf, "score": round(1.0 - 0.15 * rank, 3)})
        # Pick genes that are actually enriched in that state
        st_mask = states == st
        X_st = np.asarray(adata_rna[st_mask].layers["log1p_norm"].mean(axis=0)).ravel()
        X_other = np.asarray(adata_rna[~st_mask].layers["log1p_norm"].mean(axis=0)).ravel()
        diff = X_st - X_other
        top_g_idx = np.argsort(diff)[-8:][::-1]
        for g_i in top_g_idx:
            g_name = var_names[int(g_i)]
            if g_name not in tfs:
                gene_rows.append({"topic": k, "gene": g_name, "weight": round(float(diff[int(g_i)]), 3)})

    return topic_mat, pd.DataFrame(tf_rows), pd.DataFrame(gene_rows)


def audit_topic_collapse(
    topic_mat: np.ndarray,
    active_thresh: float = 0.10,
    collapse_warn_frac: float = 0.90,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Quantify Topic utilization, entropy, and detect mode/topic collapse red lines."""
    import numpy as np
    import pandas as pd

    n_topics = topic_mat.shape[1]
    max_act = topic_mat.max(axis=0)
    mean_act = topic_mat.mean(axis=0)
    var_act = topic_mat.var(axis=0)
    total_var = float(var_act.sum()) + 1e-12
    var_share = var_act / total_var

    # Per-cell normalized Shannon entropy over topic mixture
    p_cell = np.clip(topic_mat, 1e-12, None)
    p_cell = p_cell / p_cell.sum(axis=1, keepdims=True)
    cell_entropy = -np.sum(p_cell * np.log(p_cell), axis=1) / np.log(max(n_topics, 2))

    # Cross-topic mean utilization entropy
    p_global = np.clip(mean_act, 1e-12, None)
    p_global = p_global / p_global.sum()
    global_entropy = float(-np.sum(p_global * np.log(p_global)) / np.log(max(n_topics, 2)))

    n_active = int((max_act > active_thresh).sum())
    n_collapsed = int((max_act <= 0.05).sum())
    collapsed_frac = n_collapsed / float(n_topics)
    top2_var_share = float(np.sort(var_share)[-2:].sum()) if n_topics >= 2 else float(var_share.sum())
    collapse_red_line = bool(collapsed_frac >= collapse_warn_frac or n_active <= 2)

    diag_df = pd.DataFrame(
        {
            "topic": [f"Topic_{k}" for k in range(n_topics)],
            "mean_activity": np.round(mean_act, 5),
            "max_activity": np.round(max_act, 5),
            "variance_share": np.round(var_share, 5),
            "is_active_gt_thresh": max_act > active_thresh,
        }
    ).sort_values("mean_activity", ascending=False)

    summary = {
        "n_topics": n_topics,
        "n_active_topics": n_active,
        "n_collapsed_topics": n_collapsed,
        "collapsed_fraction": round(collapsed_frac, 4),
        "global_topic_entropy_norm": round(global_entropy, 4),
        "mean_cell_entropy_norm": round(float(cell_entropy.mean()), 4),
        "top2_topics_variance_share": round(top2_var_share, 4),
        "collapse_red_line": collapse_red_line,
    }
    return diag_df, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Sub-demo 2A Step 4: MultiVI vs scDoRI sanity checks")
    parser.add_argument("--config", type=str, default=str(SUB2A_ROOT / "configs/pbmc_demo.yaml"))
    parser.add_argument("--scdori-config", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true", help="Synthesize scDoRI outputs if not trained on GPU")
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import yaml
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    mv_path = SUB2A_ROOT / cfg["multivi"]["output_dir"] / "multivi_latent.h5ad"
    if not mv_path.exists():
        raise FileNotFoundError(f"Run 02_run_multivi_baseline.py first; missing {mv_path}")

    adata = sc.read_h5ad(mv_path)
    scdori_cfg_path = Path(args.scdori_config) if args.scdori_config else (SUB2A_ROOT / cfg["scdori"]["config_json"])

    if args.synthetic:
        topic_mat, leading_tfs, leading_genes = _generate_synthetic_scdori_outputs(adata)
        topic_obs_names = None
    else:
        topic_mat, topic_obs_names, leading_tfs, leading_genes = _extract_scdori_topics_and_grn(scdori_cfg_path, adata)

    if topic_obs_names is not None and len(topic_obs_names) == topic_mat.shape[0]:
        obs_to_idx = {obs: i for i, obs in enumerate(topic_obs_names)}
        shared_obs = [obs for obs in adata.obs_names.astype(str) if obs in obs_to_idx]
        if shared_obs:
            adata = adata[shared_obs].copy()
            topic_mat = topic_mat[[obs_to_idx[obs] for obs in shared_obs]]

    if topic_mat.shape[0] != adata.n_obs:
        n_min = min(topic_mat.shape[0], adata.n_obs)
        adata = adata[:n_min].copy()
        topic_mat = topic_mat[:n_min]

    fig_dir = SUB2A_ROOT / cfg["sanity_checks"]["figures_dir"]
    tab_dir = SUB2A_ROOT / cfg["sanity_checks"]["tables_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    # 1. UMAP & Separation Comparison
    adata.obsm["X_scdori"] = topic_mat
    adata.obsm["X_umap_multivi"] = adata.obsm["X_umap"].copy()
    sc.pp.neighbors(adata, use_rep="X_scdori", key_added="scdori")
    sc.tl.leiden(adata, neighbors_key="scdori", resolution=float(cfg["multivi"]["leiden_resolution"]), key_added="leiden_scdori")
    sc.tl.umap(adata, neighbors_key="scdori")
    adata.obsm["X_umap_scdori"] = adata.obsm["X_umap"].copy()

    celltype_col = cfg["data"].get("celltype_col", "celltype")
    color_key = celltype_col if (celltype_col in adata.obs and adata.obs[celltype_col].nunique() > 1) else "leiden_multivi"

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    adata.obsm["X_umap"] = adata.obsm["X_umap_multivi"]
    sc.pl.umap(adata, color=color_key, ax=axes[0], show=False, title="MultiVI Baseline UMAP")
    adata.obsm["X_umap"] = adata.obsm["X_umap_scdori"]
    sc.pl.umap(adata, color=color_key, ax=axes[1], show=False, title="scDoRI Topic Latent UMAP")
    fig.tight_layout()
    fig.savefig(fig_dir / "umap_multivi_vs_scdori.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    ari_clusters = float(adjusted_rand_score(adata.obs["leiden_multivi"], adata.obs["leiden_scdori"]))
    sil_mv = float(silhouette_score(adata.obsm["X_multivi"], adata.obs[color_key]))
    sil_sc = float(silhouette_score(adata.obsm["X_scdori"], adata.obs[color_key]))

    # 2. Topic Activity & Collapse Red-Line Check
    sc_cfg = cfg["sanity_checks"]
    diag_df, collapse_summary = audit_topic_collapse(
        topic_mat,
        active_thresh=float(sc_cfg["topic_active_thresh"]),
        collapse_warn_frac=float(sc_cfg["collapse_warn_frac"]),
    )
    collapse_summary["multivi_silhouette"] = round(sil_mv, 4)
    collapse_summary["scdori_silhouette"] = round(sil_sc, 4)
    collapse_summary["leiden_ari_multivi_vs_scdori"] = round(ari_clusters, 4)
    diag_df.to_csv(tab_dir / "topic_diagnostics.csv", index=False)
    (tab_dir / "sanity_summary.json").write_text(json.dumps(collapse_summary, indent=2), encoding="utf-8")

    if collapse_summary["collapse_red_line"]:
        print(
            f"[RED-LINE WARNING] scDoRI Topic collapse detected! "
            f"active_topics={collapse_summary['n_active_topics']}/{collapse_summary['n_topics']}, "
            f"top2_var_share={collapse_summary['top2_topics_variance_share']:.1%}."
        )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    axes[0].bar(range(len(diag_df)), diag_df["max_activity"], color="#2a9d8f")
    axes[0].axhline(float(sc_cfg["topic_active_thresh"]), color="#e63946", linestyle="--", label="Active threshold")
    axes[0].set_xlabel("Topic rank (sorted)")
    axes[0].set_ylabel("Max Topic Activity")
    axes[0].set_title(f"Topic Activity Distribution (Active={collapse_summary['n_active_topics']}/{collapse_summary['n_topics']})")
    axes[0].legend(fontsize=8)

    axes[1].bar(range(len(diag_df)), diag_df["variance_share"], color="#457b9d")
    axes[1].set_xlabel("Topic rank (sorted)")
    axes[1].set_ylabel("Fraction of Topic Variance")
    axes[1].set_title(f"Variance Share (Top-2 = {collapse_summary['top2_topics_variance_share']:.1%})")
    fig.tight_layout()
    fig.savefig(fig_dir / "topic_activity_diagnostics.png", dpi=180)
    plt.close(fig)

    # 3. Regulatory Self-Consistency DotPlot (Top-2 Topics -> Top TFs + Targets)
    adata.obs["dominant_topic"] = pd.Categorical([f"Topic_{k}" for k in np.argmax(topic_mat, axis=1)])
    top_k = int(sc_cfg["top_k_topics"])
    top_topic_ids = [int(t.split("_")[1]) for t in diag_df["topic"].head(top_k).tolist()]

    dot_genes: list[str] = []
    egrn_records: list[dict[str, Any]] = []
    for tid in top_topic_ids:
        tfs_t = extract_topic_tfs(leading_tfs, tid, int(sc_cfg["top_tfs_per_topic"]))
        genes_t = extract_topic_genes(leading_genes, tid, int(sc_cfg["top_genes_per_topic"]))
        for tf in tfs_t:
            if tf in adata.var_names and tf not in dot_genes:
                dot_genes.append(tf)
            egrn_records.append({"topic": f"Topic_{tid}", "role": "TF", "symbol": tf})
        for g in genes_t:
            if g in adata.var_names and g not in dot_genes:
                dot_genes.append(g)
            egrn_records.append({"topic": f"Topic_{tid}", "role": "Target", "symbol": g})

    pd.DataFrame(egrn_records).to_csv(tab_dir / "top_topics_egrn.csv", index=False)

    if dot_genes:
        adata_plot = adata.copy()
        if "log1p_norm" in adata_plot.layers:
            adata_plot.X = adata_plot.layers["log1p_norm"]
        dp = sc.pl.dotplot(
            adata_plot,
            var_names=dot_genes,
            groupby="dominant_topic",
            standard_scale="var",
            show=False,
            return_fig=True,
        )
        dp.savefig(fig_dir / "egrn_consistency_dotplot.png", dpi=180)
        plt.close("all")

    print(f"[2A Step 4] Sanity checks complete -> {fig_dir} & {tab_dir}")
    print(json.dumps(collapse_summary, indent=2))


if __name__ == "__main__":
    main()
