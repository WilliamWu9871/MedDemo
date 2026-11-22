"""Pure-Python stdlib invariant tests for MedDemo (runs without Conda/PyTorch)."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DIR1_SRC = REPO_ROOT / "direction1_pathology_multimodal_survival" / "src"
sys.path.insert(0, str(DIR1_SRC))

from direction1.utils import (
    assert_no_patient_leakage,
    encode_tsr_bins,
    extract_tcga_patient_id,
)


def _load_module(name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(name, file_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestDirection1Utils(unittest.TestCase):
    def test_extract_tcga_patient_id(self) -> None:
        self.assertEqual(
            extract_tcga_patient_id("TCGA-DD-A1EE-01Z-00-DX1.A1B2C3D4.svs"),
            "TCGA-DD-A1EE",
        )
        self.assertEqual(
            extract_tcga_patient_id("tcga-02-0047-01Z-00-DX2"),
            "TCGA-02-0047",
        )
        self.assertEqual(
            extract_tcga_patient_id("TCGA-02-0047-01Z-00-DX1"),
            extract_tcga_patient_id("TCGA-02-0047-01Z-00-DX2"),
        )
        self.assertEqual(extract_tcga_patient_id("case123-slide1.h5"), "case123")

    def test_encode_tsr_bins(self) -> None:
        cases = [
            (0.0, 0),
            (0.1249, 0),
            (0.125, 1),
            (0.188, 1),
            (0.268, 2),
            (0.392, 3),
            (1.0, 7),
        ]
        for val, expected_idx in cases:
            vec = encode_tsr_bins(val, num_bins=8)
            self.assertEqual(len(vec), 8)
            self.assertAlmostEqual(sum(vec), 1.0)
            self.assertEqual(vec.index(1.0), expected_idx, msg=f"val={val}")

    def test_assert_no_patient_leakage(self) -> None:
        assert_no_patient_leakage(["TCGA-01-0001", "TCGA-01-0002"], ["TCGA-01-0003"])
        with self.assertRaises(ValueError) as ctx:
            assert_no_patient_leakage(
                ["TCGA-01-0001", "TCGA-01-0002"],
                ["TCGA-01-0002", "TCGA-01-0003"],
                context="fold_0",
            )
        self.assertIn("TCGA-01-0002", str(ctx.exception))


class TestDirection2HelpersAndConfigs(unittest.TestCase):
    def test_normalize_peak_name(self) -> None:
        mod = _load_module(
            "prep_multiome",
            REPO_ROOT
            / "direction2_singlecell_spatial_multiomics/subdemo_2a_multiome_egrn/scripts/01_prepare_multiome_mudata.py",
        )
        self.assertEqual(mod.normalize_peak_name("chr1-100-200"), "chr1:100-200")
        self.assertEqual(mod.normalize_peak_name("chr12_1000_2500"), "chr12:1000-2500")
        self.assertEqual(mod.normalize_peak_name("chrX:500-900"), "chrX:500-900")

    def test_detect_gene_id_type(self) -> None:
        mod = _load_module(
            "dl_spatial",
            REPO_ROOT
            / "direction2_singlecell_spatial_multiomics/subdemo_2b_spatial_trajectory/scripts/01_download_spatial_and_ref.py",
        )
        self.assertEqual(
            mod.detect_gene_id_type(["ENSG00000146648.12", "ENSG00000112715", "ENSG00000141510.5"]),
            "ensembl",
        )
        self.assertEqual(
            mod.detect_gene_id_type(["EGFR", "VEGFA", "CD68", "MYT1L"]),
            "symbol",
        )
        self.assertEqual(
            mod.detect_gene_id_type(["ENSG00000146648", "EGFR", "VEGFA", "ENSG00000112715"]),
            "mixed",
        )

    def test_scdori_json_configs(self) -> None:
        cfg_dir = (
            REPO_ROOT
            / "direction2_singlecell_spatial_multiomics/subdemo_2a_multiome_egrn/configs"
        )
        for fname in ("scdori_pbmc_hg38.json", "scdori_gbm_hg38.json"):
            data = json.loads((cfg_dir / fname).read_text(encoding="utf-8"))
            for section in ("preprocessing", "training", "downstream"):
                self.assertIn(section, data, msg=f"{fname} missing {section}")
            self.assertEqual(data["preprocessing"]["genome_assembly"], "hg38")
            self.assertEqual(data["preprocessing"]["species"], "human")
            self.assertEqual(data["preprocessing"]["motif_database"], "cisbp")
            self.assertEqual(data["preprocessing"]["rna_adata_file_name"], "rna_raw.h5ad")
            self.assertEqual(data["preprocessing"]["atac_adata_file_name"], "atac_raw.h5ad")
            self.assertTrue(data["training"]["freeze_phase1_in_grn"])
            self.assertTrue(data["training"]["cache_frozen_grn"])


if __name__ == "__main__":
    unittest.main()
