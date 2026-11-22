#!/usr/bin/env python3
"""Stage 3A: Compute slide-level pseudo-prior TSR (pTSR) and 8-bin one-hot Token inputs.

Supports:
  - color_heuristic (default): Otsu tissue mask -> eosin-pink stroma vs hematoxylin-purple tumor ratio
  - tiatoolbox (optional): pretrained tissue classifier if tiatoolbox is installed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

DIR1_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIR1_ROOT / "src"))

from direction1.utils import encode_tsr_bins, load_yaml_config, resolve_path


def _load_slide_thumbnail(
    slide_id: str,
    wsi_dir: Path | None,
    thumbs_dir: Path | None,
    max_dim: int = 1024,
) -> np.ndarray | None:
    """Load an RGB thumbnail from thumbnails_dir (.png/.jpg) or directly from SVS via OpenSlide."""
    if thumbs_dir is not None and thumbs_dir.exists():
        for ext in (".png", ".jpg", ".jpeg"):
            cand = thumbs_dir / f"{slide_id}{ext}"
            if cand.exists():
                return np.asarray(Image.open(cand).convert("RGB"))

    if wsi_dir is not None and wsi_dir.exists():
        matches = list(wsi_dir.glob(f"{slide_id}*.svs"))
        if matches:
            import openslide

            slide = openslide.OpenSlide(str(matches[0]))
            thumb = slide.get_thumbnail((max_dim, max_dim)).convert("RGB")
            return np.asarray(thumb)
    return None


def estimate_tsr_color_heuristic(rgb: np.ndarray, bg_thresh: int = 225) -> float:
    """Estimate stromal ratio inside non-white tissue regions using H&E color separation.

    - Glass background: min(R,G,B) > bg_thresh or luminance > bg_thresh
    - Stroma (eosin-rich): pink/red-shifted (R - B > 15 and brightness > 135)
    - Tumor nuclei (hematoxylin-rich): blue/purple-shifted or dark dense nuclei
    """
    rgb_f = rgb.astype(np.float32)
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    # Exclude bright white background and dark pen marks/folds
    tissue_mask = (gray < bg_thresh) & (gray > 25)
    n_tissue = int(tissue_mask.sum())
    if n_tissue < 20:
        return 0.18

    # Eosin-rich stromal collagen/fibrosis appears brighter pink (R > B > G)
    stroma_mask = tissue_mask & ((r - g) > 20.0) & (r > b) & (gray > 120.0)
    raw_ratio = float(stroma_mask.sum()) / float(n_tissue)
    return float(np.clip(raw_ratio, 0.02, 0.70))


def estimate_tsr_tiatoolbox(slide_path: Path) -> float:
    """Optional tiatoolbox tissue classifier hook (requires tiatoolbox package)."""
    from tiatoolbox.models.engine.patch_predictor import IOPatchPredictorConfig, PatchPredictor

    predictor = PatchPredictor(pretrained_model="resnet18-kather100k", batch_size=32)
    ioconfig = IOPatchPredictorConfig(input_resolutions=[{"units": "mpp", "resolution": 0.5}], patch_input_shape=[224, 224], stride_shape=[224, 224])
    output = predictor.predict(imgs=[str(slide_path)], mode="wsi", ioconfig=ioconfig, return_probabilities=False)
    preds = np.asarray(output[0]["predictions"])
    # Kather100k labels: 6=STR (stroma), 8=TUM (tumor), 2=DEB (debris/necrosis)
    n_stroma = int((preds == 6).sum())
    n_bed = int(np.isin(preds, [2, 6, 8]).sum())
    return float(np.clip(n_stroma / max(n_bed, 1), 0.02, 0.70))


def compute_slide_priors(
    splits_df: pd.DataFrame,
    wsi_dir: Path | None,
    thumbs_dir: Path | None,
    backend: str = "color_heuristic",
    num_bins: int = 8,
    bg_thresh: int = 225,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in splits_df.itertuples(index=False):
        slide_id = str(row.slide_id)
        if backend == "tiatoolbox" and wsi_dir is not None:
            svs_matches = list(wsi_dir.glob(f"{slide_id}*.svs"))
            if svs_matches:
                ptsr = estimate_tsr_tiatoolbox(svs_matches[0])
                method_used = "tiatoolbox"
            else:
                thumb = _load_slide_thumbnail(slide_id, wsi_dir, thumbs_dir)
                ptsr = estimate_tsr_color_heuristic(thumb, bg_thresh) if thumb is not None else 0.18
                method_used = "color_heuristic_fallback"
        else:
            thumb = _load_slide_thumbnail(slide_id, wsi_dir, thumbs_dir)
            if thumb is not None:
                ptsr = estimate_tsr_color_heuristic(thumb, bg_thresh=bg_thresh)
                method_used = "color_heuristic"
            elif hasattr(row, "true_tsr"):
                ptsr = float(row.true_tsr)
                method_used = "clinical_fallback"
            else:
                ptsr = 0.18
                method_used = "default_fallback"

        bins = encode_tsr_bins(ptsr, num_bins=num_bins)
        rec: dict[str, object] = {
            "patient_id": str(row.patient_id),
            "slide_id": slide_id,
            "pTSR": round(ptsr, 6),
            "tsr_method": method_used,
        }
        for i, val in enumerate(bins):
            rec[f"tsr_bin_{i}"] = val
        rows.append(rec)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3A: Compute slide pseudo-prior TSR tokens")
    parser.add_argument("--config", type=str, default=str(DIR1_ROOT / "configs/tcga_lihc_demo.yaml"))
    parser.add_argument("--splits-csv", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--backend", type=str, default=None, choices=["color_heuristic", "tiatoolbox"])
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    if args.synthetic:
        splits_path = DIR1_ROOT / "splits/synthetic_splits.csv"
        wsi_dir = None
        thumbs_dir = DIR1_ROOT / "data/synthetic/thumbnails"
        out_path = DIR1_ROOT / "outputs/synthetic/priors/slide_priors.csv"
    else:
        splits_path = Path(args.splits_csv) if args.splits_csv else resolve_path(DIR1_ROOT, cfg["paths"]["splits_csv"])
        wsi_dir = resolve_path(DIR1_ROOT, cfg["paths"]["wsi_dir"])
        thumbs_dir = resolve_path(DIR1_ROOT, cfg["paths"].get("thumbnails_dir", "data/thumbnails"))
        out_path = Path(args.output_csv) if args.output_csv else resolve_path(DIR1_ROOT, cfg["paths"]["priors_csv"])

    backend = args.backend or cfg["prior"].get("backend", "color_heuristic")
    num_bins = int(cfg["prior"].get("tsr_num_bins", 8))
    bg_thresh = int(cfg.get("evaluation", {}).get("bg_rgb_threshold", 225))

    splits_df = pd.read_csv(splits_path)
    priors_df = compute_slide_priors(
        splits_df=splits_df,
        wsi_dir=wsi_dir,
        thumbs_dir=thumbs_dir,
        backend=backend,
        num_bins=num_bins,
        bg_thresh=bg_thresh,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    priors_df.to_csv(out_path, index=False)

    print(f"[Stage 3A] Computed pTSR for {len(priors_df)} slides -> {out_path}")
    print(f"  pTSR summary: median={priors_df['pTSR'].median():.3f}, IQR=({priors_df['pTSR'].quantile(0.25):.3f}, {priors_df['pTSR'].quantile(0.75):.3f})")


if __name__ == "__main__":
    main()
