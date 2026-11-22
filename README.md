# MedDemo: Two-Direction Experimental Demo Suite

本仓库围绕目录中的三篇前沿生物医学论文，按照**数据形态、环境依赖与验证标准完全解耦**的原则，构建了两套独立的最小闭环实验路径（均适配单张 **16–24 GB 显存 GPU**，且内置 `--synthetic` 合成数据烟雾测试模式以便快速跑通代码逻辑）。

---

## 论文与开源技术栈映射

| 论文文件 | 所属方向 | 官方 / 参考开源仓库 | 公开数据源 |
| --- | --- | --- | --- |
| [Token-guided multimodal prognosis in hepatocellular carcinom.pdf](./Token-guided%20multimodal%20prognosis%20in%20hepatocellular%20carcinom.pdf) (*Huang et al., Gut 2026*) | **方向一**：计算病理与先验引导多模态生存预测 | `mahmoodlab/TRIDENT` (WSI 分割/切块/`conch_v15` 特征) + `mahmoodlab/SurvPath` (`GatedABMIL` + 4-bin `NLLSurvLoss`) | GDC Portal (`TCGA-LIHC` / `TCGA-GBM` Diagnostic Slide) |
| [Decoding the regulatory programs governing glioblastoma plasticity.pdf](./Decoding%20the%20regulatory%20programs%20governing%20glioblastoma%20plasticity.pdf) (*Saraswat et al., Nature 2026*) | **方向二 · Sub-demo 2A**：单细胞双组学调控推断 (eGRN) | `bioFAM/scDoRI` (v2.1+) + `PMBio/GBM_analysis` + `scvi-tools` (`MultiVI`) | 10x PBMC 3k/10k Multiome；GBM Atlas (`gbmspace.org`, `E-MTAB-17183`, Zenodo `21873381`) |
| [A spatiotemporal cancer cell trajectory underlies glioblastoma heterogeneity.pdf](./A%20spatiotemporal%20cancer%20cell%20trajectory%20underlies%20glioblastoma%20heterogeneity.pdf) (*de Jong et al., Nature 2026*) | **方向二 · Sub-demo 2B**：空间转录组解卷积与组织轨迹映射 | `BayraktarLab/GBM_space_paper` + `BayraktarLab/cell2location` + `Squidpy` + `PMBio/spaceTree` (可选) | `MahmoodLab/hest` (HuggingFace 按器官/癌种按需拉取) + `E-MTAB-17164` / `GSE237183` / `Neftel SCP393` |

---

## 目录结构与两个独立实验路径

```text
MedDemo/
├── direction1_pathology_multimodal_survival/   # 方向一：WSI MIL + Prior Token 生存预测 (独立环境 meddemo_patho)
│   ├── env/environment.yml
│   ├── configs/                                # tcga_lihc_demo.yaml, tcga_gbm_demo.yaml
│   ├── src/direction1/                         # 患者防泄漏工具、SurvPath NLLSurvLoss、ABMIL & Token-Guided Cross-Attention
│   └── scripts/                                # 00 下载 -> 01 患者划分 -> 02 TRIDENT -> 03 伪先验TSR -> 04 消融训练 -> 05 验收出图
│
├── direction2_singlecell_spatial_multiomics/   # 方向二：单细胞多组学 + 空间转录组映射 (独立环境 meddemo_sc_spatial)
│   ├── env/environment.yml
│   ├── scripts/setup_repos.sh                  # 一键拉取 scDoRI, GBM_analysis, GBM_space_paper
│   ├── subdemo_2a_multiome_egrn/               # 2A: MuData QC -> MultiVI 基线 -> 官方 scDoRI Notebook -> 塌缩/调控核验
│   └── subdemo_2b_spatial_trajectory/          # 2B: HEST按需拉取 -> 基因名对齐&Moran's I -> cell2location -> Niche & 伪时间投影
│
└── tests/
    └── test_core_invariants.py                 # 核心不变式单元测试 (纯 Python 标准库即可运行)
```

---

## 环境隔离与快速入口

> [!IMPORTANT]
> 请务必为两个方向分别创建独立的 Conda 环境，避免 `TRIDENT`/`OpenSlide` 与 `scverse`/`cell2location`/`bedtools` 的底层 C++/NumPy/CUDA 依赖冲突。

### 方向一快速开始
请见 [direction1_pathology_multimodal_survival/README.md](./direction1_pathology_multimodal_survival/README.md)：
```bash
conda env create -f direction1_pathology_multimodal_survival/env/environment.yml
conda activate meddemo_patho
bash direction1_pathology_multimodal_survival/scripts/setup_repos.sh
```

### 方向二快速开始
请见 [direction2_singlecell_spatial_multiomics/README.md](./direction2_singlecell_spatial_multiomics/README.md)：
```bash
conda env create -f direction2_singlecell_spatial_multiomics/env/environment.yml
conda activate meddemo_sc_spatial
bash direction2_singlecell_spatial_multiomics/scripts/setup_repos.sh
```

### 离线逻辑自检（无需 GPU/Conda）
```bash
/usr/bin/python3 -m unittest discover -s tests -v
```

---

## 核心交付产物与失败红线核查

| 维度 | 方向一 (`direction1_pathology_multimodal_survival`) | 方向二 (`direction2_singlecell_spatial_multiomics`) |
| --- | --- | --- |
| **核心交付产物** | 1. 5 折 `C-index` 消融对比表 (`Exp 1.0/1.1/1.2`)<br>2. KM 生存曲线 + 多因素 Cox 表 + RCS 非线性曲线<br>3. WSI 高/低风险 Attention 热力图叠加图 | 1. `MultiVI` vs `scDoRI` 联合 UMAP 与 Top Topic 调控 DotPlot<br>2. `cell2location` 叠加在 H&E 上的细胞丰度与 Niche 图<br>3. 扩散伪时间轨迹 (`DPT`) 的二维组织空间投影图 |
| **自动化红线审计** | 1. **防泄漏红线**：强制按 12 位 TCGA `patient_id` 分组 `StratifiedGroupKFold`，且若 `train C-index > 0.92` 而 `val C-index < 0.55` 自动报警<br>2. **背景噪点红线**：自动统计 Top-10% Attention Patch 落在空白玻璃区的比例 | 1. **基因 ID 对齐红线**：2B 中若 scRNA-seq 与 Visium 交集基因 `< 8000` 自动报错并诊断 Ensembl vs Symbol 不一致<br>2. **Topic 塌缩红线**：2A 自动检测是否 `>=90%` Topic 趋近于 0<br>3. **空间噪点红线**：2B 检查解卷积结果的 Moran's I 防止椒盐噪点 |
