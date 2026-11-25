"""Direction 1 package: computational pathology MIL and prior-guided multimodal survival."""

from direction1.utils import (
    assert_no_patient_leakage,
    encode_tsr_bins,
    extract_tcga_patient_id,
    infer_patch_step_lv0,
    load_yaml_config,
)

__all__ = [
    "extract_tcga_patient_id",
    "encode_tsr_bins",
    "assert_no_patient_leakage",
    "infer_patch_step_lv0",
    "load_yaml_config",
    "NLLSurvLoss",
    "fit_survival_bins",
    "transform_survival_bins",
    "hazards_to_survival_and_risk",
    "PatientMILDataset",
    "GatedABMIL",
    "TokenGuidedMultimodalSurv",
]


def __getattr__(name: str):
    if name in {
        "NLLSurvLoss",
        "fit_survival_bins",
        "transform_survival_bins",
        "hazards_to_survival_and_risk",
    }:
        from direction1 import losses

        return getattr(losses, name)
    if name == "PatientMILDataset":
        from direction1 import dataset

        return getattr(dataset, name)
    if name in {"GatedABMIL", "TokenGuidedMultimodalSurv"}:
        from direction1 import models

        return getattr(models, name)
    raise AttributeError(f"module 'direction1' has no attribute {name!r}")
