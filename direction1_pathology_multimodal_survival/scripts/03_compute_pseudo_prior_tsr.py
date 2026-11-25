#!/usr/bin/env python3
"""Stage 3A: Compute slide-level pseudo-prior TSR (pTSR) and 8-bin one-hot Token inputs.

Supports:
  - kather_probe (default for real slides): train/cache a 9-class tissue probe on public
    Kather-100K (DykeF/NCTCRCHE100K) using the active TRIDENT foundation encoder, then infer
    pTSR = N_STR / (N_TUM + N_STR + N_DEB) directly from pre-extracted .h5 embeddings.
  - color_heuristic (default for --synthetic & skipped-slide fallback): Ruifrok & Johnston H&E
    optical-density stain deconvolution on RGB thumbnails.
  - tiatoolbox (optional): pretrained tissue classifier if tiatoolbox is installed.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

DIR1_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIR1_ROOT / "src"))

from direction1.dataset import (
    _load_trident_h5,
    build_h5_index,
    is_valid_trident_h5,
    resolve_h5_from_index,
)
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
    """Estimate stromal ratio inside non-white tissue regions via H&E optical-density separation.

    Uses the Ruifrok & Johnston (2001) H&E stain deconvolution matrix so eosin-rich fibrous
    stroma is separated from hematoxylin-rich tumor nuclei and pink hepatocyte cytoplasm.
    """
    rgb_f = rgb.astype(np.float32)
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    # Exclude bright white background and dark pen marks/folds
    tissue_mask = (gray < bg_thresh) & (gray > 25)
    n_tissue = int(tissue_mask.sum())
    if n_tissue < 20:
        return 0.18

    od = -np.log10(np.clip(rgb_f, 1.0, 255.0) / 255.0)
    # Ruifrok & Johnston (2001) standard H&E stain matrix rows: [Hematoxylin, Eosin, Residual]
    he_matrix = np.array(
        [
            [0.650, 0.704, 0.286],
            [0.072, 0.990, 0.105],
            [0.268, 0.570, 0.776],
        ],
        dtype=np.float32,
    )
    he_inv = np.linalg.inv(he_matrix)
    stains = od @ he_inv
    hematoxylin = stains[..., 0]
    eosin = stains[..., 1]

    # Fibrous/collagenous stroma: clear eosin signal with low hematoxylin (sparse nuclei)
    stroma_mask = tissue_mask & (eosin > 0.06) & (hematoxylin < 0.06) & ((eosin - hematoxylin) > 0.07)
    raw_ratio = float(stroma_mask.sum()) / float(n_tissue)
    return float(np.clip(raw_ratio, 0.02, 0.95))


def estimate_tsr_tiatoolbox(slide_path: Path) -> float:
    """Optional tiatoolbox tissue classifier hook (requires tiatoolbox package)."""
    from tiatoolbox.models.engine.patch_predictor import IOPatchPredictorConfig, PatchPredictor

    predictor = PatchPredictor(pretrained_model="resnet18-kather100k", batch_size=32)
    ioconfig = IOPatchPredictorConfig(
        input_resolutions=[{"units": "mpp", "resolution": 0.5}],
        patch_input_shape=[224, 224],
        stride_shape=[224, 224],
    )
    output = predictor.predict(imgs=[str(slide_path)], mode="wsi", ioconfig=ioconfig, return_probabilities=False)
    preds = np.asarray(output[0]["predictions"])
    # Kather100k labels: 6=STR (stroma), 8=TUM (tumor), 2=DEB (debris/necrosis)
    n_stroma = int((preds == 6).sum())
    n_bed = int(np.isin(preds, [2, 6, 8]).sum())
    return float(np.clip(n_stroma / max(n_bed, 1), 0.02, 0.70))


def train_or_load_kather_probe(
    probe_path: Path,
    encoder_name: str,
    dataset_repo: str = "DykeF/NCTCRCHE100K",
    max_per_class: int = 400,
    batch_size: int = 256,
    seed: int = 42,
) -> dict[str, Any]:
    """Train or load a 9-class Kather-100K linear probe on top of TRIDENT patch embeddings."""
    if probe_path.exists():
        with np.load(probe_path, allow_pickle=False) as d:
            probe = {
                "scaler_mean": d["scaler_mean"].astype(np.float32),
                "scaler_scale": d["scaler_scale"].astype(np.float32),
                "coef": d["coef"].astype(np.float32),
                "intercept": d["intercept"].astype(np.float32),
                "classes": [str(c) for c in d["classes"].tolist()],
                "val_acc": float(d["val_acc"][0]),
            }
        print(f"[Stage 3A] Loaded cached Kather-100K probe ({encoder_name}, val_acc={probe['val_acc']:.4f}) <- {probe_path}")
        return probe

    trident_dir = DIR1_ROOT / "external/TRIDENT"
    if trident_dir.exists() and str(trident_dir) not in sys.path:
        sys.path.insert(0, str(trident_dir))

    import pyarrow.parquet as pq
    import torch
    from huggingface_hub import hf_hub_download
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from trident.patch_encoder_models.load import encoder_factory

    shard_files = (
        "default/partial-validation/0000.parquet",
        "default/partial-validation/0001.parquet",
    )
    all_images: list[Any] = []
    all_labels: list[str] = []
    for shard_rel in shard_files:
        local_pq = hf_hub_download(
            repo_id=dataset_repo,
            filename=shard_rel,
            repo_type="dataset",
            revision="refs/convert/parquet",
        )
        tbl = pq.read_table(local_pq, columns=["image", "label"])
        all_images.extend(tbl["image"].to_pylist())
        all_labels.extend(str(lbl) for lbl in tbl["label"].to_pylist())

    by_class: dict[str, list[int]] = {}
    for idx, lbl in enumerate(all_labels):
        by_class.setdefault(lbl, []).append(idx)

    rng = np.random.default_rng(seed)
    sampled_indices: list[int] = []
    for lbl in sorted(by_class):
        idxs = np.asarray(by_class[lbl], dtype=np.int64)
        if len(idxs) > max_per_class:
            idxs = rng.choice(idxs, size=max_per_class, replace=False)
        sampled_indices.extend(int(i) for i in idxs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    encoder = encoder_factory(encoder_name).to(device)
    encoder.eval()
    transforms = encoder.eval_transforms
    precision = getattr(encoder, "precision", torch.float32)
    if device.type == "cuda" and precision == torch.float32:
        precision = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_amp = device.type == "cuda" and precision in (torch.float16, torch.bfloat16)

    feats_chunks: list[np.ndarray] = []
    y_sampled: list[str] = []
    batch_tensors: list[torch.Tensor] = []

    print(
        f"[Stage 3A] Extracting {encoder_name} embeddings on {len(sampled_indices)} Kather-100K patches "
        f"({len(by_class)} classes, device={device})...",
        flush=True,
    )
    with torch.no_grad():
        for pos, idx in enumerate(sampled_indices):
            entry = all_images[idx]
            raw_bytes = entry["bytes"] if isinstance(entry, dict) else entry
            pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            batch_tensors.append(transforms(pil_img))
            y_sampled.append(all_labels[idx])

            if len(batch_tensors) == batch_size or (pos + 1) == len(sampled_indices):
                xb = torch.stack(batch_tensors, dim=0).to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=precision, enabled=use_amp):
                    emb = encoder(xb)
                feats_chunks.append(emb.detach().float().cpu().numpy())
                batch_tensors.clear()

    X = np.concatenate(feats_chunks, axis=0).astype(np.float32)
    y = np.asarray(y_sampled)

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    val_scaler = StandardScaler()
    X_tr_z = val_scaler.fit_transform(X_tr)
    X_va_z = val_scaler.transform(X_va)
    val_clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500, solver="lbfgs", random_state=seed)
    val_clf.fit(X_tr_z, y_tr)
    va_pred = val_clf.predict(X_va_z)
    val_acc = float(accuracy_score(y_va, va_pred))
    macro_f1 = float(f1_score(y_va, va_pred, average="macro"))

    recalls = []
    for cls_name in ("STR", "TUM", "DEB"):
        mask = y_va == cls_name
        if mask.any():
            recalls.append(f"{cls_name}={float((va_pred[mask] == cls_name).mean()):.3f}")
    print(
        f"[Stage 3A] Trained {encoder_name} Kather-100K 9-class tissue probe: "
        f"val_acc={val_acc:.4f}, macro_f1={macro_f1:.4f} ({', '.join(recalls)})",
        flush=True,
    )

    full_scaler = StandardScaler()
    X_full_z = full_scaler.fit_transform(X)
    full_clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500, solver="lbfgs", random_state=seed)
    full_clf.fit(X_full_z, y)

    probe_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        probe_path,
        scaler_mean=full_scaler.mean_.astype(np.float32),
        scaler_scale=full_scaler.scale_.astype(np.float32),
        coef=full_clf.coef_.astype(np.float32),
        intercept=full_clf.intercept_.astype(np.float32),
        classes=np.asarray(full_clf.classes_, dtype="U16"),
        val_acc=np.asarray([val_acc], dtype=np.float32),
    )

    return {
        "scaler_mean": full_scaler.mean_.astype(np.float32),
        "scaler_scale": full_scaler.scale_.astype(np.float32),
        "coef": full_clf.coef_.astype(np.float32),
        "intercept": full_clf.intercept_.astype(np.float32),
        "classes": [str(c) for c in full_clf.classes_.tolist()],
        "val_acc": val_acc,
    }


def estimate_tsr_from_h5_probe(h5_path: Path, probe: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """Classify pre-extracted TRIDENT .h5 patch embeddings and compute tumor-bed pTSR."""
    feats, _ = _load_trident_h5(h5_path)
    expected_dim = int(probe["coef"].shape[1])
    if feats.shape[1] != expected_dim:
        raise ValueError(
            f"Feature dimension mismatch for {h5_path}: got {feats.shape[1]}, expected {expected_dim} from probe"
        )

    z = (feats - probe["scaler_mean"]) / np.maximum(probe["scaler_scale"], 1e-8)
    logits = z @ probe["coef"].T + probe["intercept"]
    logits -= logits.max(axis=1, keepdims=True)
    exp_l = np.exp(logits)
    probs = exp_l / np.maximum(exp_l.sum(axis=1, keepdims=True), 1e-12)
    pred_idx = np.argmax(probs, axis=1)

    cls_to_idx = {name: i for i, name in enumerate(probe["classes"])}
    idx_tum = cls_to_idx["TUM"]
    idx_str = cls_to_idx["STR"]
    idx_deb = cls_to_idx["DEB"]
    idx_norm = cls_to_idx.get("NORM")
    idx_lym = cls_to_idx.get("LYM")

    n_tum = int((pred_idx == idx_tum).sum())
    n_str = int((pred_idx == idx_str).sum())
    n_deb = int((pred_idx == idx_deb).sum())
    n_norm = int((pred_idx == idx_norm).sum()) if idx_norm is not None else 0
    n_lym = int((pred_idx == idx_lym).sum()) if idx_lym is not None else 0

    n_bed = n_tum + n_str + n_deb
    n_patches = int(len(feats))
    tumor_bed_frac = n_bed / max(n_patches, 1)

    if n_bed >= 10:
        raw_tsr = n_str / float(n_bed)
    else:
        p_str = float(probs[:, idx_str].sum())
        p_bed = float((probs[:, idx_tum] + probs[:, idx_str] + probs[:, idx_deb]).sum())
        raw_tsr = p_str / max(p_bed, 1e-6)

    ptsr = float(np.clip(raw_tsr, 0.01, 0.98))
    stats = {
        "n_patches": n_patches,
        "n_tum": n_tum,
        "n_str": n_str,
        "n_deb": n_deb,
        "n_norm": n_norm,
        "n_lym": n_lym,
        "tumor_bed_frac": round(float(tumor_bed_frac), 4),
    }
    return ptsr, stats


def compute_slide_priors(
    splits_df: pd.DataFrame,
    wsi_dir: Path | None,
    thumbs_dir: Path | None,
    backend: str = "color_heuristic",
    num_bins: int = 8,
    bg_thresh: int = 225,
    feats_dir: Path | None = None,
    patch_encoder: str = "hibou_l",
    kather_dataset: str = "DykeF/NCTCRCHE100K",
    max_samples_per_class: int = 400,
    probe_path: Path | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    probe: dict[str, Any] | None = None
    exact_h5: dict[str, Path] = {}
    prefix_h5: dict[str, Path] = {}

    if backend == "kather_probe":
        if feats_dir is not None and feats_dir.exists():
            exact_h5, prefix_h5 = build_h5_index(feats_dir)
        if exact_h5:
            resolved_probe = probe_path or (DIR1_ROOT / f"outputs/kather100k_{patch_encoder}_probe.npz")
            probe = train_or_load_kather_probe(
                probe_path=resolved_probe,
                encoder_name=patch_encoder,
                dataset_repo=kather_dataset,
                max_per_class=max_samples_per_class,
                seed=seed,
            )
        else:
            print(
                f"[Stage 3A][WARN] No .h5 features found under {feats_dir}; falling back to color_heuristic.",
                flush=True,
            )

    rows: list[dict[str, object]] = []
    for row in splits_df.itertuples(index=False):
        slide_id = str(row.slide_id)
        stats: dict[str, Any] = {
            "n_patches": 0,
            "n_tum": 0,
            "n_str": 0,
            "n_deb": 0,
            "n_norm": 0,
            "n_lym": 0,
            "tumor_bed_frac": np.nan,
        }

        if backend == "kather_probe" and probe is not None:
            h5_path = resolve_h5_from_index(exact_h5, prefix_h5, slide_id)
            if h5_path is not None and is_valid_trident_h5(h5_path):
                ptsr, stats = estimate_tsr_from_h5_probe(h5_path, probe)
                method_used = f"kather_probe_{patch_encoder}"
            else:
                thumb = _load_slide_thumbnail(slide_id, wsi_dir, thumbs_dir)
                ptsr = estimate_tsr_color_heuristic(thumb, bg_thresh) if thumb is not None else 0.18
                method_used = "color_heuristic_fallback"
        elif backend == "tiatoolbox" and wsi_dir is not None:
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
            **stats,
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
    parser.add_argument("--feats-dir", type=str, default=None, help="Directory with TRIDENT .h5 features")
    parser.add_argument("--patch-encoder", type=str, default=None, help="TRIDENT encoder name for Kather-100K probe")
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["kather_probe", "color_heuristic", "tiatoolbox"],
    )
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    if args.synthetic:
        splits_path = DIR1_ROOT / "splits/synthetic_splits.csv"
        wsi_dir = None
        thumbs_dir = DIR1_ROOT / "data/synthetic/thumbnails"
        out_path = DIR1_ROOT / "outputs/synthetic/priors/slide_priors.csv"
        feats_dir = None
        backend = "color_heuristic"
    else:
        splits_path = Path(args.splits_csv) if args.splits_csv else resolve_path(DIR1_ROOT, cfg["paths"]["splits_csv"])
        wsi_dir = resolve_path(DIR1_ROOT, cfg["paths"]["wsi_dir"])
        thumbs_dir = resolve_path(DIR1_ROOT, cfg["paths"].get("thumbnails_dir", "data/thumbnails"))
        out_path = Path(args.output_csv) if args.output_csv else resolve_path(DIR1_ROOT, cfg["paths"]["priors_csv"])
        feats_dir = (
            Path(args.feats_dir)
            if args.feats_dir
            else resolve_path(DIR1_ROOT, cfg["paths"]["trident_job_dir"])
        )
        backend = args.backend or cfg["prior"].get("backend", "kather_probe")

    patch_encoder = args.patch_encoder or cfg.get("trident", {}).get("patch_encoder", "hibou_l")
    kather_dataset = cfg["prior"].get("kather_dataset", "DykeF/NCTCRCHE100K")
    max_per_class = int(cfg["prior"].get("max_samples_per_class", 400))
    num_bins = int(cfg["prior"].get("tsr_num_bins", 8))
    bg_thresh = int(cfg.get("evaluation", {}).get("bg_rgb_threshold", 225))
    seed = int(cfg.get("splits", {}).get("default_seed", 42))
    probe_path = out_path.parent / f"kather100k_{patch_encoder}_probe.npz"

    splits_df = pd.read_csv(splits_path)
    priors_df = compute_slide_priors(
        splits_df=splits_df,
        wsi_dir=wsi_dir,
        thumbs_dir=thumbs_dir,
        backend=backend,
        num_bins=num_bins,
        bg_thresh=bg_thresh,
        feats_dir=feats_dir,
        patch_encoder=patch_encoder,
        kather_dataset=kather_dataset,
        max_samples_per_class=max_per_class,
        probe_path=probe_path,
        seed=seed,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    priors_df.to_csv(out_path, index=False)

    print(f"[Stage 3A] Computed pTSR ({backend}) for {len(priors_df)} slides -> {out_path}")
    print(
        f"  pTSR summary: median={priors_df['pTSR'].median():.3f}, "
        f"IQR=({priors_df['pTSR'].quantile(0.25):.3f}, {priors_df['pTSR'].quantile(0.75):.3f})"
    )


if __name__ == "__main__":
    main()
