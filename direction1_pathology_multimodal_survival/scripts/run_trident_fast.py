#!/usr/bin/env python3
"""Accelerated TRIDENT runner enabling Tensor Core AMP, VRAM-aware batching, and safe resume.

Upstream TRIDENT hardcodes Hibou-L (`HibouLInferenceEncoder`) to ``precision = torch.float32``
and calls ``batch_features.cpu().numpy()`` directly in ``WSI.extract_patch_features()``.
This wrapper:
  1. Enables TF32 matmul/cuDNN on NVIDIA Ampere+ GPUs.
  2. Enables ``bfloat16`` (fallback ``float16``) autocast for ``HibouLInferenceEncoder`` and
     casts its output embeddings back to ``float32`` so ``.numpy()`` and ``.h5`` remain compatible.
  3. Cleans any half-written/corrupted ``.h5`` feature files from an interrupted run before resume.
  4. Auto-scales ``--feat_batch_size`` from available GPU VRAM if not explicitly passed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DIR1_ROOT = Path(__file__).resolve().parents[1]
TRIDENT_DIR = DIR1_ROOT / "external" / "TRIDENT"


def _cleanup_corrupt_h5(job_dir: Path) -> None:
    """Remove unreadable or incomplete feature .h5 files left behind by an interrupted run."""
    if not job_dir.exists():
        return
    import h5py

    for pattern in ("**/features_*/*.h5", "**/patch_features_*/*.h5"):
        for h5_path in sorted(job_dir.glob(pattern)):
            try:
                with h5py.File(h5_path, "r") as hf:
                    if "features" not in hf and "feats" not in hf:
                        raise KeyError("missing features dataset")
                    ds = hf["features"] if "features" in hf else hf["feats"]
                    _ = ds.shape
            except Exception as exc:
                print(
                    f"[Stage 2][Resume] Removing incomplete feature file {h5_path.name} ({exc})",
                    flush=True,
                )
                try:
                    h5_path.unlink()
                except OSError:
                    pass


def _patch_hibou_amp() -> None:
    """Enable bfloat16/float16 autocast on HibouLInferenceEncoder with float32 output cast."""
    if os.environ.get("TRIDENT_FORCE_FP32") == "1":
        return

    import torch
    from trident.patch_encoder_models.load import HibouLInferenceEncoder

    if not torch.cuda.is_available():
        return

    orig_build = HibouLInferenceEncoder._build
    orig_forward = HibouLInferenceEncoder.forward

    def fast_build(self, *args, **kwargs):
        model, eval_transform, precision = orig_build(self, *args, **kwargs)
        if torch.cuda.is_available() and precision == torch.float32:
            precision = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            print(f"[Stage 2][FastAMP] Enabled {precision} Tensor Core autocast for hibou_l", flush=True)
        return model, eval_transform, precision

    def fast_forward(self, x):
        out = orig_forward(self, x)
        return out.float() if isinstance(out, torch.Tensor) else out

    HibouLInferenceEncoder._build = fast_build
    HibouLInferenceEncoder.forward = fast_forward


def main() -> None:
    """Configure GPU speedups, clean interrupted artifacts, and invoke TRIDENT."""
    if str(TRIDENT_DIR) not in sys.path:
        sys.path.insert(0, str(TRIDENT_DIR))

    import torch

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        if "--feat_batch_size" not in sys.argv[1:]:
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024**3)
            feat_bs = 512 if vram_gb >= 35.0 else (256 if vram_gb >= 20.0 else 128)
            sys.argv.extend(["--feat_batch_size", str(feat_bs)])
            print(
                f"[Stage 2][FastAMP] GPU={props.name} ({vram_gb:.1f} GB VRAM) -> auto --feat_batch_size {feat_bs}",
                flush=True,
            )

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--job_dir", type=str, default=None)
    known, _ = parser.parse_known_args(sys.argv[1:])
    if known.job_dir:
        _cleanup_corrupt_h5(Path(known.job_dir))

    _patch_hibou_amp()

    import run_batch_of_slides

    run_batch_of_slides.main()


if __name__ == "__main__":
    main()
