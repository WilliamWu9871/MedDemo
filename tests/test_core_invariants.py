"""Pure-Python stdlib invariant tests for MedDemo (runs without Conda/PyTorch)."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DIR1_SRC = REPO_ROOT / "direction1_pathology_multimodal_survival" / "src"
sys.path.insert(0, str(DIR1_SRC))

from direction1.utils import (
    assert_no_patient_leakage,
    encode_tsr_bins,
    extract_tcga_patient_id,
    infer_patch_step_lv0,
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

    def test_infer_patch_step_lv0(self) -> None:
        self.assertEqual(
            infer_patch_step_lv0([(0, 0), (224, 0), (448, 0), (0, 224)]),
            224,
        )
        self.assertEqual(
            infer_patch_step_lv0([(0, 0), (448, 0), (1344, 0), (0, 448), (448, 448)]),
            448,
        )
        self.assertEqual(
            infer_patch_step_lv0([(0, 0), (512, 0), (0, 512)]),
            512,
        )
        self.assertEqual(infer_patch_step_lv0([(100, 200)], default_step=448), 448)


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
        sub2a_root = REPO_ROOT / "direction2_singlecell_spatial_multiomics/subdemo_2a_multiome_egrn"
        cfg_dir = sub2a_root / "configs"
        ext_motif_dir = (
            REPO_ROOT / "direction2_singlecell_spatial_multiomics/external/scDoRI/assets/motif_database"
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
            # Relative paths in scDoRI JSON resolve against config's parent directory (configs/)
            self.assertEqual(
                (cfg_dir / data["preprocessing"]["data_dir"]).resolve(),
                (sub2a_root / "data/processed").resolve(),
            )
            self.assertEqual(
                (cfg_dir / data["preprocessing"]["motif_directory"]).resolve(),
                ext_motif_dir.resolve(),
            )
            self.assertEqual(
                (cfg_dir / data["training"]["weights_folder_scdori"]).resolve(),
                (sub2a_root / "outputs/scdori/phase1").resolve(),
            )
            self.assertEqual(
                (cfg_dir / data["training"]["weights_folder_grn"]).resolve(),
                (sub2a_root / "outputs/scdori/phase2").resolve(),
            )
            self.assertEqual(
                (cfg_dir / data["downstream"]["grn_output"]).resolve(),
                (sub2a_root / "outputs/scdori/downstream").resolve(),
            )

    def test_sanity_check_extractors(self) -> None:
        mod = _load_module(
            "sanity_cmp",
            REPO_ROOT
            / "direction2_singlecell_spatial_multiomics/subdemo_2a_multiome_egrn/scripts/04_sanity_check_compare.py",
        )
        official_tfs = [
            {"topic": "Topic_2", "tf": "SOX2", "minmax_consensus": 0.91},
            {"topic": "Topic_2", "tf": "OLIG2", "minmax_consensus": 0.85},
            {"topic": "Topic_3", "tf": "TCF7", "minmax_consensus": 0.80},
        ]
        official_genes = [
            {"Topic": "Topic_2", "Leading genes": "SOX2, OLIG2, EGFR, DLL3, PDGFRA"},
            {"Topic": "Topic_3", "Leading genes": "TCF7, LEF1"},
        ]
        self.assertEqual(mod.extract_topic_tfs(official_tfs, 2, top_n=3), ["SOX2", "OLIG2"])
        self.assertEqual(mod.extract_topic_genes(official_genes, 2, top_n=3), ["SOX2", "OLIG2", "EGFR"])

        syn_tfs = [
            {"topic": 2, "tf": "PAX5", "score": 1.0},
            {"topic": 2, "tf": "EBF1", "score": 0.85},
        ]
        syn_genes = [
            {"topic": 2, "gene": "CD19", "weight": 2.1},
            {"topic": 2, "gene": "MS4A1", "weight": 1.8},
        ]
        self.assertEqual(mod.extract_topic_tfs(syn_tfs, 2, top_n=2), ["PAX5", "EBF1"])
        self.assertEqual(mod.extract_topic_genes(syn_genes, 2, top_n=2), ["CD19", "MS4A1"])

    def test_patch_kernel_json(self) -> None:
        mod = _load_module(
            "patch_ks",
            REPO_ROOT / "direction2_singlecell_spatial_multiomics/scripts/patch_kernelspec.py",
        )
        with tempfile.TemporaryDirectory() as tmp:
            kj = Path(tmp) / "kernel.json"
            kj.write_text(
                json.dumps(
                    {
                        "argv": [
                            "/content/venv_meddemo_dir2_py311/bin/python",
                            "-m",
                            "ipykernel_launcher",
                            "-f",
                            "{connection_file}",
                        ],
                        "display_name": "Python 3.11 (meddemo_dir2)",
                        "language": "python",
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(mod.patch_kernel_json(kj))
            patched = json.loads(kj.read_text(encoding="utf-8"))
            self.assertEqual(patched["argv"][0], "/content/venv_meddemo_dir2_py311/bin/python")
            self.assertEqual(patched["argv"][1], "-c")
            self.assertIn("SYSTEM_CONFIG_DIRS", patched["argv"][2])
            self.assertEqual(patched["argv"][3:], ["-f", "{connection_file}"])
            self.assertFalse(mod.patch_kernel_json(kj))


if __name__ == "__main__":
    unittest.main()
