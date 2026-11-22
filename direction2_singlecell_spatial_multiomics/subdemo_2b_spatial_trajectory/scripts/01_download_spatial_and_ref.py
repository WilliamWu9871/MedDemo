#!/usr/bin/env python3
"""Sub-demo 2B Step 1: Selectively download HEST-1k brain Visium samples or prepare local/synthetic data.

Uses lightweight `huggingface_hub` pattern downloads to avoid downloading >2 TB of HEST-1k.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

SUB2B_ROOT = Path(__file__).resolve().parents[1]

_ENSEMBL_RE = re.compile(r"^ENS[A-Z]*G\d{6,}(?:\.\d+)?$", re.IGNORECASE)


def detect_gene_id_type(names: Iterable[str]) -> str:
    """Classify a collection of gene identifiers as 'ensembl', 'symbol', or 'mixed'."""
    name_list = [str(x).strip() for x in names if str(x).strip()]
    if not name_list:
        return "symbol"
    n_ens = sum(1 for x in name_list if _ENSEMBL_RE.match(x))
    frac = n_ens / float(len(name_list))
    if frac >= 0.80:
        return "ensembl"
    if frac <= 0.10:
        return "symbol"
    return "mixed"


def _discover_hest_metadata_csv(repo_id: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    csvs = sorted(f for f in files if re.match(r"^HEST_v[\d_]+\.csv$", Path(f).name))
    return csvs[-1] if csvs else "HEST_v1_3_0.csv"


def _attach_thumbnail_if_missing(adata, thumb_path: Path | None, sample_id: str) -> None:
    import numpy as np
    from PIL import Image

    if "spatial" in adata.uns and isinstance(adata.uns["spatial"], dict) and adata.uns["spatial"]:
        lib_id = next(iter(adata.uns["spatial"]))
        if adata.uns["spatial"][lib_id].get("images"):
            return

    if thumb_path is not None and thumb_path.exists():
        img = np.asarray(Image.open(thumb_path).convert("RGB")) / 255.0
        adata.uns["spatial"] = {
            sample_id: {
                "images": {"lowres": img, "hires": img},
                "scalefactors": {
                    "tissue_lowres_scalef": 1.0,
                    "tissue_hires_scalef": 1.0,
                    "spot_diameter_fullres": 20.0,
                },
            }
        }


def download_hest_subset(cfg: dict, raw_dir: Path, proc_dir: Path) -> list[Path]:
    import pandas as pd
    import scanpy as sc
    from huggingface_hub import hf_hub_download, snapshot_download

    q = cfg["hest_query"]
    repo_id = q["repo_id"]
    meta_name = _discover_hest_metadata_csv(repo_id)
    print(f"[2B Step 1] Fetching metadata table {meta_name} from {repo_id}...")
    meta_local = hf_hub_download(repo_id=repo_id, filename=meta_name, repo_type="dataset", local_dir=raw_dir / "hest")
    meta_df = pd.read_csv(meta_local)

    sub = meta_df.copy()
    if "organ" in sub.columns:
        sub = sub[sub["organ"].astype(str).str.lower() == str(q["organ"]).lower()]
    if "st_technology" in sub.columns:
        sub = sub[sub["st_technology"].astype(str).str.lower().str.contains(str(q["st_technology"]).lower())]

    if "oncotree_code" in sub.columns:
        pref = [c.upper() for c in q.get("preferred_oncotree", [])]
        sub["_pref_rank"] = sub["oncotree_code"].astype(str).str.upper().map(lambda c: 0 if c in pref else 1)
        sub = sub.sort_values(["_pref_rank", "id"])

    max_samples = int(q.get("max_samples", 3))
    chosen = sub.head(max_samples)
    if chosen.empty:
        raise RuntimeError("No matching Brain Visium samples found in HEST metadata.")

    sample_ids = chosen["id"].astype(str).tolist()
    print(f"[2B Step 1] Selected HEST samples: {sample_ids}")

    patterns = []
    for sid in sample_ids:
        patterns.extend([f"st/{sid}.h5ad", f"thumbnails/{sid}*", f"metadata/{sid}*", f"spatial_plots/{sid}*"])
        if q.get("download_full_wsi", False):
            patterns.append(f"wsis/{sid}*")

    hest_dir = raw_dir / "hest"
    snapshot_download(repo_id=repo_id, repo_type="dataset", allow_patterns=patterns, local_dir=hest_dir)

    saved_paths: list[Path] = []
    for idx, sid in enumerate(sample_ids):
        st_path = hest_dir / "st" / f"{sid}.h5ad"
        if not st_path.exists():
            continue
        adata = sc.read_h5ad(st_path)
        adata.obs["sample"] = sid
        thumbs = list((hest_dir / "thumbnails").glob(f"{sid}*"))
        _attach_thumbnail_if_missing(adata, thumbs[0] if thumbs else None, sid)
        out_p = proc_dir / ("visium_sample.h5ad" if idx == 0 else f"visium_sample_{sid}.h5ad")
        adata.write_h5ad(out_p)
        saved_paths.append(out_p)
    return saved_paths


def prepare_reference_h5ad(
    ref_path: Path,
    out_path: Path,
    max_cells: int,
    cell_state_col: str,
    batch_col: str,
    seed: int = 42,
) -> Path:
    import numpy as np
    import scanpy as sc
    import scipy.sparse as sp

    adata_ref = sc.read_h5ad(ref_path)
    if cell_state_col not in adata_ref.obs:
        raise KeyError(f"Reference AnnData missing cell_state_col={cell_state_col!r}; found {list(adata_ref.obs.columns)}")
    if batch_col not in adata_ref.obs:
        adata_ref.obs[batch_col] = "ref_batch_1"

    if adata_ref.n_obs > max_cells:
        rng = np.random.default_rng(seed)
        frac = max_cells / float(adata_ref.n_obs)
        chosen = []
        for _, grp in adata_ref.obs.groupby(cell_state_col, observed=True):
            n_take = max(5, int(round(len(grp) * frac)))
            chosen.extend(rng.choice(grp.index.to_numpy(), size=min(len(grp), n_take), replace=False).tolist())
        adata_ref = adata_ref[chosen[:max_cells]].copy()

    if not sp.issparse(adata_ref.X):
        adata_ref.X = sp.csr_matrix(adata_ref.X)
    adata_ref.write_h5ad(out_path)
    return out_path


def generate_synthetic_spatial_and_ref(proc_dir: Path, seed: int = 42) -> tuple[Path, Path]:
    """Generate synthetic GBM Visium slide (with H&E tissue image) and scRNA-seq reference."""
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    states = [
        "Dev_OPC_like",
        "AC_progenitor_like",
        "AC_gliosis_like",
        "Hypoxic",
        "Myeloid_TAM",
        "Vascular",
    ]
    state_markers = {
        "Dev_OPC_like": ["OLIG1", "SOX2", "PDGFRA", "DLL3"],
        "AC_progenitor_like": ["EGFR", "GFAP", "SOX9", "CLU"],
        "AC_gliosis_like": ["VIM", "CD44", "CHI3L1", "TIMP1"],
        "Hypoxic": ["VEGFA", "HILPDA", "CA9", "LDHA"],
        "Myeloid_TAM": ["CD68", "CD163", "CSF1R", "TREM2"],
        "Vascular": ["PECAM1", "VWF", "CLDN5", "FLT1"],
    }
    marker_list = [g for gs in state_markers.values() for g in gs]
    other_genes = [f"GENE{i:04d}" for i in range(1, 227)]
    genes = marker_list + other_genes  # 250 genes
    n_genes = len(genes)

    # 1. Synthetic scRNA-seq reference (1,800 cells across 6 GBM states)
    n_per_state = 300
    n_cells = n_per_state * len(states)
    ref_X = rng.poisson(0.5, size=(n_cells, n_genes)).astype(np.float32)
    ref_states, ref_batches = [], []
    state_profiles = {}
    for s_idx, st in enumerate(states):
        c0, c1 = s_idx * n_per_state, (s_idx + 1) * n_per_state
        prof = np.full(n_genes, 0.5, dtype=np.float32)
        for mg in state_markers[st]:
            prof[genes.index(mg)] = 10.0
        block_start = 24 + s_idx * 35
        prof[block_start : block_start + 35] = 4.5
        state_profiles[st] = prof
        ref_X[c0:c1] = rng.poisson(prof[None, :], size=(n_per_state, n_genes)).astype(np.float32)
        ref_states.extend([st] * n_per_state)
        ref_batches.extend([f"donor_{(i % 3) + 1}" for i in range(n_per_state)])

    adata_ref = ad.AnnData(
        X=sp.csr_matrix(ref_X),
        obs=pd.DataFrame(
            {
                "cell_state": pd.Categorical(ref_states),
                "sample": pd.Categorical(ref_batches),
            },
            index=[f"REF_{i:05d}" for i in range(n_cells)],
        ),
        var=pd.DataFrame(index=genes),
    )
    ref_out = proc_dir / "reference_sc.h5ad"
    adata_ref.write_h5ad(ref_out)

    # 2. Synthetic Visium 25x25 tissue grid with spatial core-to-periphery trajectory + synthetic H&E
    grid_n = 25
    img_px = 500
    step_px = img_px / float(grid_n)
    he_img = np.full((img_px, img_px, 3), 0.96, dtype=np.float32)

    coords = []
    spot_counts = []
    true_props = []
    spot_ids = []

    for gy in range(grid_n):
        for gx in range(grid_n):
            cx = (gx + 0.5) * step_px
            cy = (gy + 0.5) * step_px
            # Normalized radius from necrotic center (0 = necrotic core, 1 = cortical edge)
            r = np.hypot((gx - 12.0) / 11.0, (gy - 12.0) / 11.0)
            if r > 1.05:
                continue  # outside tissue slice

            # Smooth spatial weights matching Paper 3: Hypoxic core -> Gliosis -> Progenitor -> Dev_OPC edge
            w_hyp = np.exp(-((r - 0.15) ** 2) / (2 * 0.16**2)) * 9.0
            w_gli = np.exp(-((r - 0.45) ** 2) / (2 * 0.18**2)) * 8.0
            w_tam = np.exp(-((r - 0.35) ** 2) / (2 * 0.22**2)) * 4.5
            w_pro = np.exp(-((r - 0.72) ** 2) / (2 * 0.18**2)) * 8.0
            w_dev = np.exp(-((r - 0.92) ** 2) / (2 * 0.18**2)) * 8.0
            w_vas = (0.8 + 2.5 * np.exp(-((r - 0.80) ** 2) / (2 * 0.15**2)))
            w = np.asarray([w_dev, w_pro, w_gli, w_hyp, w_tam, w_vas], dtype=np.float32)
            p = w / w.sum()
            n_cells_spot = 30.0
            expected_expr = np.zeros(n_genes, dtype=np.float32)
            for st, prop in zip(states, p):
                expected_expr += (prop * n_cells_spot) * state_profiles[st] * 0.35

            spot_counts.append(rng.poisson(expected_expr).astype(np.float32))
            coords.append([cx, cy])
            true_props.append(p)
            spot_ids.append(f"SPOT_{gx:02d}_{gy:02d}")

            # Paint matching H&E color into thumbnail (pale pink core necrosis, dense purple rim, lighter cortex)
            y0, y1 = int(gy * step_px), int((gy + 1) * step_px)
            x0, x1 = int(gx * step_px), int((gx + 1) * step_px)
            if r < 0.28:
                col = np.array([0.88, 0.72, 0.78], dtype=np.float32)  # necrotic/hypoxic eosinophilic
            elif r < 0.65:
                col = np.array([0.55, 0.34, 0.66], dtype=np.float32)  # dense cellular core/gliosis
            else:
                col = np.array([0.74, 0.56, 0.78], dtype=np.float32)  # infiltrative margin
            he_img[y0:y1, x0:x1] = np.clip(col + rng.normal(0, 0.03, size=3), 0.1, 0.98)

    adata_vis = ad.AnnData(
        X=sp.csr_matrix(np.asarray(spot_counts, dtype=np.float32)),
        obs=pd.DataFrame(
            {"sample": "SYNTHETIC_GBM_VISIUM", "in_tissue": 1},
            index=spot_ids,
        ),
        var=pd.DataFrame(index=genes),
    )
    adata_vis.obsm["spatial"] = np.asarray(coords, dtype=np.float32)
    adata_vis.obsm["true_proportions"] = np.asarray(true_props, dtype=np.float32)
    adata_vis.uns["spatial"] = {
        "SYNTHETIC_GBM_VISIUM": {
            "images": {"hires": he_img, "lowres": he_img},
            "scalefactors": {
                "tissue_hires_scalef": 1.0,
                "tissue_lowres_scalef": 1.0,
                "spot_diameter_fullres": step_px * 0.8,
            },
        }
    }
    vis_out = proc_dir / "visium_sample.h5ad"
    adata_vis.write_h5ad(vis_out)
    return vis_out, ref_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Sub-demo 2B Step 1: Acquire spatial Visium & scRNA-seq reference")
    parser.add_argument("--config", type=str, default=str(SUB2B_ROOT / "configs/gbm_spatial_demo.yaml"))
    parser.add_argument("--source", type=str, default="hest", choices=["hest", "local", "synthetic"])
    parser.add_argument("--visium-h5ad", type=str, default=None, help="Local Visium .h5ad when --source local")
    parser.add_argument("--ref-h5ad", type=str, default=None, help="Local scRNA-seq reference .h5ad")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import yaml

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    raw_dir = SUB2B_ROOT / cfg["paths"]["raw_dir"]
    proc_dir = SUB2B_ROOT / cfg["paths"]["processed_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "synthetic":
        vis_p, ref_p = generate_synthetic_spatial_and_ref(proc_dir, seed=args.seed)
        print(f"[2B Step 1][SYNTHETIC] Saved {vis_p} and {ref_p}")
        return

    if args.source == "hest":
        saved = download_hest_subset(cfg, raw_dir, proc_dir)
        print(f"[2B Step 1][HEST] Saved {len(saved)} Visium sample(s) under {proc_dir}")
    elif args.source == "local":
        if not args.visium_h5ad:
            raise ValueError("--visium-h5ad is required when --source local")
        import scanpy as sc

        adata_vis = sc.read_h5ad(args.visium_h5ad)
        if "sample" not in adata_vis.obs:
            adata_vis.obs["sample"] = Path(args.visium_h5ad).stem
        adata_vis.write_h5ad(proc_dir / "visium_sample.h5ad")
        print(f"[2B Step 1][LOCAL] Copied Visium -> {proc_dir / 'visium_sample.h5ad'}")

    if args.ref_h5ad:
        ref_cfg = cfg["reference"]
        out_ref = prepare_reference_h5ad(
            ref_path=Path(args.ref_h5ad),
            out_path=proc_dir / "reference_sc.h5ad",
            max_cells=int(ref_cfg["max_cells"]),
            cell_state_col=ref_cfg["cell_state_col"],
            batch_col=ref_cfg["batch_col"],
            seed=args.seed,
        )
        print(f"[2B Step 1] Prepared stratified scRNA-seq reference -> {out_ref}")
    else:
        print(
            "[2B Step 1] Note: Pass --ref-h5ad /path/to/gbmap_or_neftel.h5ad to prepare reference_sc.h5ad "
            "before running Step 2/3."
        )


if __name__ == "__main__":
    main()
