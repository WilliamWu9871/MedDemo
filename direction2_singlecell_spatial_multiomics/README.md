# 方向二：单细胞多组学调控推断与空间轨迹映射 Demo（对应 Paper 2 & 3）

本目录覆盖两篇密切衔接的 Nature 2026 论文：
1. **Sub-demo 2A（`subdemo_2a_multiome_egrn/`）**：对应 [Decoding the regulatory programs governing glioblastoma plasticity.pdf](../Decoding%20the%20regulatory%20programs%20governing%20glioblastoma%20plasticity.pdf)（*Saraswat et al., Nature 2026*），基于 `scvi-tools` (`MultiVI`) 基线与官方 `bioFAM/scDoRI`（v2.0+）推断单细胞增强子驱动基因调控网络（eGRN）及 Topic 活性；并与 `PMBio/GBM_analysis` 的可塑性及 TAP/TRS 分析脚本衔接。
2. **Sub-demo 2B（`subdemo_2b_spatial_trajectory/`）**：对应 [A spatiotemporal cancer cell trajectory underlies glioblastoma heterogeneity.pdf](../A%20spatiotemporal%20cancer%20cell%20trajectory%20underlies%20glioblastoma%20heterogeneity.pdf)（*de Jong et al., Nature 2026*），复用 `MahmoodLab/hest`（HuggingFace 按器官/技术按需轻量化拉取，避免下载 >2 TB 全量库）、`Squidpy` 空间自相关、`BayraktarLab/cell2location` 空间解卷积、`BayraktarLab/GBM_space_paper` NMF 组织微环境（TM）识别及扩散伪时间（`DPT` / 可选 `scFates`）空间轨迹投射。

---

## 1. 独立 Conda 环境、Google Colab 与官方仓库准备

* **Google Colab Pro (`L4` + `High-RAM`) 一键启动**：直接上传并打开 [colab_launcher.ipynb](./colab_launcher.ipynb)（内置 `bedtools` 与固定版本 `scverse`/`scDoRI` 安装、Google Drive 实时日志同步与断点备份）。

> [!IMPORTANT]
> 请使用独立环境 `meddemo_sc_spatial`，切勿与方向一的 `TRIDENT` / `OpenSlide` / `pyvips` 混用。基础环境也不直接安装重依赖 `hest` 包，而是直接通过轻量的 `huggingface_hub` 按样品 ID 精准拉取 HEST-1k 脑肿瘤 Visium 子集，避免 C++/NumPy 冲突。

```bash
conda env create -f direction2_singlecell_spatial_multiomics/env/environment.yml
conda activate meddemo_sc_spatial

# 浅克隆官方仓库 bioFAM/scDoRI, PMBio/GBM_analysis, BayraktarLab/GBM_space_paper 并安装 scDoRI
bash direction2_singlecell_spatial_multiomics/scripts/setup_repos.sh

# （可选）若需要克隆 PMBio/spaceTree 进行联合克隆与细胞状态图神经网络映射：
# bash direction2_singlecell_spatial_multiomics/scripts/setup_repos.sh --with-spacetree
```

---

## 2. 实验子模块入口

- **Sub-demo 2A（配对 RNA+ATAC 联合表征与 eGRN 推断）**：详见 [subdemo_2a_multiome_egrn/README.md](./subdemo_2a_multiome_egrn/README.md)
- **Sub-demo 2B（单细胞图谱向空间转录组与 H&E 映射）**：详见 [subdemo_2b_spatial_trajectory/README.md](./subdemo_2b_spatial_trajectory/README.md)
