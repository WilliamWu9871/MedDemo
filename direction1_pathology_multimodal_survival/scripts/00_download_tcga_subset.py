#!/usr/bin/env python3
"""Stage 0: Query/download a TCGA diagnostic WSI subset via GDC API, or generate synthetic demo data.

Only Open Access Diagnostic Slides (FFPE, -DX) are selected; frozen Tissue Slides (-TS)
are strictly excluded per computational pathology best practice.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DIR1_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIR1_ROOT / "src"))

from direction1.utils import extract_tcga_patient_id, load_yaml_config, resolve_path

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"


def _parse_stage_numeric(stage_str: Any) -> float:
    if not isinstance(stage_str, str):
        return 1.0
    s = stage_str.upper()
    if "IV" in s:
        return 4.0
    if "III" in s:
        return 3.0
    if "II" in s:
        return 2.0
    if "I" in s:
        return 1.0
    return 1.0


def query_gdc_diagnostic_slides(
    project_id: str,
    n_patients: int,
    min_os_days: int = 1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Query GDC API for Open Access SVS Diagnostic Slides and matched OS clinical fields."""
    import requests

    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [project_id]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}},
            {"op": "in", "content": {"field": "experimental_strategy", "value": ["Diagnostic Slide"]}},
            {"op": "in", "content": {"field": "access", "value": ["open"]}},
            {"op": "in", "content": {"field": "data_format", "value": ["SVS"]}},
        ],
    }
    fields = ",".join(
        [
            "file_id",
            "file_name",
            "file_size",
            "md5sum",
            "cases.submitter_id",
            "cases.demographic.vital_status",
            "cases.demographic.days_to_death",
            "cases.demographic.age_at_index",
            "cases.demographic.gender",
            "cases.diagnoses.days_to_last_follow_up",
            "cases.diagnoses.ajcc_pathologic_stage",
            "cases.samples.sample_type",
        ]
    )
    params = {
        "filters": json.dumps(filters),
        "fields": fields,
        "format": "JSON",
        "size": "2000",
    }
    resp = requests.get(GDC_FILES_ENDPOINT, params=params, timeout=60)
    resp.raise_for_status()
    hits = resp.json().get("data", {}).get("hits", [])

    rows: list[dict[str, Any]] = []
    for hit in hits:
        file_name = str(hit.get("file_name", ""))
        stem = Path(file_name).stem
        # Strictly exclude frozen Tissue Slides (-TS) and keep Diagnostic Slides (-DX)
        if "-TS" in stem.upper() or "-DX" not in stem.upper():
            continue

        cases = hit.get("cases") or []
        if not cases:
            continue
        case = cases[0]
        raw_patient = str(case.get("submitter_id") or extract_tcga_patient_id(stem))
        patient_id = extract_tcga_patient_id(raw_patient)

        demo = case.get("demographic") or {}
        diags = case.get("diagnoses") or [{}]
        diag = diags[0] if diags else {}

        vital = str(demo.get("vital_status") or "").strip().lower()
        if vital == "dead":
            os_event = 1
            os_time = demo.get("days_to_death")
        elif vital == "alive":
            os_event = 0
            os_time = diag.get("days_to_last_follow_up")
        else:
            continue

        if os_time is None:
            continue
        try:
            os_time_val = float(os_time)
        except (TypeError, ValueError):
            continue
        if os_time_val < min_os_days:
            continue

        age_val = demo.get("age_at_index")
        age = float(age_val) if age_val is not None else np.nan
        gender = str(demo.get("gender") or "unknown").strip().lower()
        sex = 1.0 if gender == "male" else 0.0
        stage_raw = str(diag.get("ajcc_pathologic_stage") or "Stage I")
        stage = _parse_stage_numeric(stage_raw)

        rows.append(
            {
                "file_id": hit["file_id"],
                "file_name": file_name,
                "slide_id": stem,
                "patient_id": patient_id,
                "file_size": int(hit.get("file_size", 0)),
                "md5sum": hit.get("md5sum", ""),
                "os_time": os_time_val,
                "os_event": os_event,
                "age": age,
                "sex": sex,
                "sex_label": gender,
                "stage": stage,
                "stage_label": stage_raw,
                "project": project_id,
            }
        )

    full_df = pd.DataFrame(rows)
    if full_df.empty:
        raise RuntimeError(f"No eligible open-access Diagnostic SVS slides found for {project_id}.")

    # Impute missing ages with median
    if full_df["age"].isna().any():
        full_df["age"] = full_df["age"].fillna(full_df["age"].median())

    # Patient-level stratified selection
    rng = np.random.default_rng(seed)
    patient_df = full_df.groupby("patient_id", as_index=False).first()[["patient_id", "os_event"]]
    if len(patient_df) > n_patients:
        event_patients = patient_df[patient_df["os_event"] == 1]["patient_id"].to_numpy()
        cens_patients = patient_df[patient_df["os_event"] == 0]["patient_id"].to_numpy()
        event_frac = len(event_patients) / len(patient_df)
        n_event = min(len(event_patients), max(1, int(round(n_patients * event_frac))))
        n_cens = min(len(cens_patients), n_patients - n_event)
        chosen = np.concatenate(
            [
                rng.choice(event_patients, size=n_event, replace=False),
                rng.choice(cens_patients, size=n_cens, replace=False),
            ]
        )
        selected_df = full_df[full_df["patient_id"].isin(set(chosen))].copy()
    else:
        selected_df = full_df.copy()

    selected_df = selected_df.sort_values(["patient_id", "slide_id"]).reset_index(drop=True)
    manifest_df = selected_df[["file_id", "file_name", "md5sum", "file_size"]].rename(
        columns={"file_id": "id", "file_name": "filename", "md5sum": "md5", "file_size": "size"}
    )
    manifest_df["state"] = "released"
    return selected_df, manifest_df


def download_gdc_files(
    manifest_df: pd.DataFrame,
    wsi_dir: Path,
    use_gdc_client: bool = False,
    manifest_path: Path | None = None,
) -> None:
    """Download SVS files using gdc-client or resumable HTTP streaming."""
    import requests
    from tqdm import tqdm

    wsi_dir.mkdir(parents=True, exist_ok=True)
    if use_gdc_client and shutil.which("gdc-client") and manifest_path is not None:
        cmd = ["gdc-client", "download", "-m", str(manifest_path), "-d", str(wsi_dir)]
        print("[download] Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        return

    for row in manifest_df.itertuples(index=False):
        dest = wsi_dir / row.filename
        if dest.exists() and dest.stat().st_size == int(row.size):
            continue
        url = f"{GDC_DATA_ENDPOINT}/{row.id}"
        print(f"[download] Streaming {row.filename} ({row.size / 1e6:.1f} MB)...")
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            tmp = dest.with_suffix(".part")
            with tmp.open("wb") as handle, tqdm(
                total=int(row.size), unit="B", unit_scale=True, desc=row.filename
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                        pbar.update(len(chunk))
            tmp.rename(dest)


def generate_synthetic_cohort(
    out_root: Path,
    n_patients: int = 72,
    seed: int = 42,
) -> Path:
    """Generate a realistic synthetic TCGA-style pathology cohort with .h5 features and thumbnails.

    Includes multi-slide patients and Paper 1's inverted-U TSR hazard pattern peaking near 0.268,
    allowing end-to-end validation of Stages 1, 3, and 4 on CPU or single GPU in seconds.
    """
    import h5py
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(seed)
    meta_dir = out_root / "metadata"
    feats_dir = out_root / "trident_feats" / "features"
    thumbs_dir = out_root / "thumbnails"
    meta_dir.mkdir(parents=True, exist_ok=True)
    feats_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    feat_dim = 64
    grid_n = 16
    patch_px = 256
    thumb_tile = 16  # 16 * 16 = 256px thumbnail

    # Prototypes for tissue classes (tumor, stroma, background)
    w_tumor = rng.normal(0, 1, size=feat_dim).astype(np.float32)
    w_stroma = rng.normal(0, 1, size=feat_dim).astype(np.float32)
    w_stroma -= np.dot(w_stroma, w_tumor) / (np.dot(w_tumor, w_tumor) + 1e-8) * w_tumor
    w_stroma /= np.linalg.norm(w_stroma) + 1e-8
    w_tumor /= np.linalg.norm(w_tumor) + 1e-8
    w_risk_signal = rng.normal(0, 1, size=feat_dim).astype(np.float32)
    w_risk_signal /= np.linalg.norm(w_risk_signal) + 1e-8

    clinical_rows: list[dict[str, Any]] = []
    for p_idx in range(n_patients):
        patient_id = f"TCGA-SY-{1000 + p_idx:04d}"
        age = float(np.clip(rng.normal(59.0, 9.5), 32.0, 84.0))
        sex = int(rng.random() < 0.75)
        stage = float(rng.choice([1.0, 2.0, 3.0], p=[0.52, 0.32, 0.16]))

        # Simulate TSR distribution matching Paper 1 (median ~0.18, range 0.04..0.52)
        true_tsr = float(np.clip(rng.beta(2.2, 8.5) * 0.85 + 0.03, 0.04, 0.52))
        # Paper 1 inverted-U hazard peaking near TSR ~0.268 + clinical stage effect
        tsr_hazard = 1.6 * np.exp(-((true_tsr - 0.268) ** 2) / (2 * 0.095**2))
        latent_risk = 0.55 * (stage - 1.5) + 0.02 * (age - 59.0) + tsr_hazard + rng.normal(0, 0.25)

        rate = 0.00055 * np.exp(latent_risk)
        raw_time = float(rng.exponential(1.0 / rate))
        censor_time = float(rng.uniform(450.0, 2200.0))
        os_time = float(max(15.0, round(min(raw_time, censor_time), 1)))
        os_event = int(raw_time <= censor_time)

        # Give ~17% of patients 2 diagnostic slides to test patient-level aggregation & leak checks
        n_slides = 2 if (p_idx % 6 == 0) else 1
        for s_idx in range(n_slides):
            slide_id = f"{patient_id}-01Z-00-DX{s_idx + 1}"
            slide_tsr = float(np.clip(true_tsr + rng.normal(0.0, 0.015), 0.03, 0.55))

            img = Image.new("RGB", (grid_n * thumb_tile, grid_n * thumb_tile), (245, 245, 245))
            draw = ImageDraw.Draw(img)

            coords_list: list[list[int]] = []
            feats_list: list[np.ndarray] = []

            for gy in range(grid_n):
                for gx in range(grid_n):
                    # Circular/elliptical tissue region with clear glass margins on edges
                    dist = np.hypot((gx - 7.5) / 6.2, (gy - 7.5) / 6.0)
                    if dist > 1.05:
                        continue  # background excluded by segmentation
                    if dist > 0.92 and rng.random() < 0.35:
                        continue

                    x0, y0 = gx * thumb_tile, gy * thumb_tile
                    is_stroma = rng.random() < slide_tsr
                    if is_stroma:
                        # Eosin-rich pink stroma
                        color = (
                            int(np.clip(225 + rng.integers(-12, 12), 180, 245)),
                            int(np.clip(145 + rng.integers(-15, 15), 110, 185)),
                            int(np.clip(175 + rng.integers(-15, 15), 140, 210)),
                        )
                        vec = 1.4 * w_stroma + (0.9 * tsr_hazard) * w_risk_signal
                    else:
                        # Hematoxylin-rich purple/blue tumor parenchyma
                        color = (
                            int(np.clip(125 + rng.integers(-18, 18), 80, 165)),
                            int(np.clip(75 + rng.integers(-15, 15), 45, 115)),
                            int(np.clip(155 + rng.integers(-18, 18), 115, 195)),
                        )
                        vec = 1.4 * w_tumor + (0.35 * (stage - 1.5)) * w_risk_signal

                    draw.rectangle([x0, y0, x0 + thumb_tile - 1, y0 + thumb_tile - 1], fill=color)
                    noise = rng.normal(0.0, 0.35, size=feat_dim).astype(np.float32)
                    feats_list.append((vec + noise).astype(np.float32))
                    coords_list.append([gx * patch_px, gy * patch_px])

            img.save(thumbs_dir / f"{slide_id}.png")
            with h5py.File(feats_dir / f"{slide_id}.h5", "w") as hf:
                hf.create_dataset("features", data=np.asarray(feats_list, dtype=np.float32))
                hf.create_dataset("coords", data=np.asarray(coords_list, dtype=np.int64))

            clinical_rows.append(
                {
                    "patient_id": patient_id,
                    "slide_id": slide_id,
                    "file_name": f"{slide_id}.svs",
                    "os_time": os_time,
                    "os_event": os_event,
                    "age": round(age, 1),
                    "sex": sex,
                    "sex_label": "male" if sex == 1 else "female",
                    "stage": stage,
                    "stage_label": f"Stage {'I' * int(stage)}",
                    "true_tsr": round(true_tsr, 4),
                    "project": "SYNTHETIC-LIHC",
                }
            )

    clinical_path = meta_dir / "clinical.csv"
    pd.DataFrame(clinical_rows).to_csv(clinical_path, index=False)
    return clinical_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 0: TCGA subset metadata/download or synthetic cohort")
    parser.add_argument("--config", type=str, default=str(DIR1_ROOT / "configs/tcga_lihc_demo.yaml"))
    parser.add_argument("--n-patients", type=int, default=None, help="Override patient count")
    parser.add_argument("--download", action="store_true", help="Download SVS files in addition to manifest/clinical")
    parser.add_argument("--use-gdc-client", action="store_true", help="Prefer gdc-client CLI over HTTP streaming")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic dry-run cohort instead of GDC query")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    n_patients = args.n_patients or int(cfg["experiment"]["n_patients"])

    if args.synthetic:
        syn_root = DIR1_ROOT / "data" / "synthetic"
        clinical_path = generate_synthetic_cohort(syn_root, n_patients=n_patients, seed=args.seed)
        df = pd.read_csv(clinical_path)
        print(
            f"[Stage 0][SYNTHETIC] Generated {df['patient_id'].nunique()} patients "
            f"({len(df)} slides) -> {clinical_path}"
        )
        return

    project_id = cfg["experiment"]["project"]
    min_os_days = int(cfg["experiment"].get("min_os_days", 1))
    meta_dir = resolve_path(DIR1_ROOT, cfg["paths"]["metadata_dir"])
    wsi_dir = resolve_path(DIR1_ROOT, cfg["paths"]["wsi_dir"])
    clinical_path = resolve_path(DIR1_ROOT, cfg["paths"]["clinical_csv"])
    meta_dir.mkdir(parents=True, exist_ok=True)

    clinical_df, manifest_df = query_gdc_diagnostic_slides(
        project_id=project_id,
        n_patients=n_patients,
        min_os_days=min_os_days,
        seed=args.seed,
    )
    manifest_path = meta_dir / "gdc_manifest.tsv"
    manifest_df.to_csv(manifest_path, sep="\t", index=False)
    clinical_df.to_csv(clinical_path, index=False)

    total_gb = manifest_df["size"].sum() / (1024**3)
    print(
        f"[Stage 0][{project_id}] Selected {clinical_df['patient_id'].nunique()} patients "
        f"({len(clinical_df)} Diagnostic SVS slides, ~{total_gb:.2f} GB)."
    )
    print(f"  Manifest: {manifest_path}")
    print(f"  Clinical: {clinical_path}")

    if args.download:
        download_gdc_files(
            manifest_df=manifest_df,
            wsi_dir=wsi_dir,
            use_gdc_client=args.use_gdc_client,
            manifest_path=manifest_path,
        )
    else:
        print("  Pass --download to fetch SVS files, or run: gdc-client download -m", manifest_path, "-d", wsi_dir)


if __name__ == "__main__":
    main()
