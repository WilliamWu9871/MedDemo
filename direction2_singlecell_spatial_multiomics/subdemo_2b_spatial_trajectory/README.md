# Sub-demo 2B：单细胞图谱向空间转录组与 H&E 映射（对应 Paper 3）

对应论文：[A spatiotemporal cancer cell trajectory underlies glioblastoma heterogeneity.pdf](../../A%20spatiotemporal%20cancer%20cell%20trajectory%20underlies%20glioblastoma%20heterogeneity.pdf)（*de Jong et al., Nature 2026*）。  
核心复用仓库：`BayraktarLab/GBM_space_paper`（`code/2.SPATIAL_TMs/` 与 `code/5.CELLULAR_TRAJECTORIES/`）、`BayraktarLab/cell2location`、`Squidpy`、`MahmoodLab/hest`（HuggingFace 轻量化按需拉取，避开 >2 TB 全量下载）。

---

## 1. 快速合成闭环测试（含合成 H&E、6 种 GBM 细胞状态与核心-边缘伪时间梯度）

```bash
cd direction2_singlecell_spatial_multiomics/subdemo_2b_spatial_trajectory

python scripts/01_download_spatial_and_ref.py --source synthetic
python scripts/02_spatial_qc_and_squidpy.py --synthetic
python scripts/03_run_cell2location_deconv.py --synthetic
python scripts/04_spatial_niches_and_trajectory.py
```

---

## 2. 真实 HEST Brain Visium + 单细胞参考集（Neftel SCP393 / GBmap）端到端流程

### Step 1：从 HuggingFace `MahmoodLab/hest` 按器官/癌种精准拉取 2–3 例脑部 Visium
先确保已登录 HuggingFace（需在 `https://huggingface.co/datasets/MahmoodLab/hest` 同意协议后 `huggingface-cli login`），并准备好下采样到 ≤1 万细胞的单细胞参考集（含 `cell_state` 标签）：
```bash
cd direction2_singlecell_spatial_multiomics/subdemo_2b_spatial_trajectory

python scripts/01_download_spatial_and_ref.py \
  --config configs/gbm_spatial_demo.yaml \
  --source hest \
  --ref-h5ad /path/to/neftel_or_gbmap_raw_counts.h5ad
```
> 若改用 Paper 3 官方 ArrayExpress `E-MTAB-17164` 或 `GSE237183` Visium `.h5ad`，请改用 `--source local --visium-h5ad ... --ref-h5ad ...`。

### Step 2：基因命名对齐红线校验（Symbol vs Ensembl）与 `Squidpy` 空间自相关
```bash
python scripts/02_spatial_qc_and_squidpy.py --config configs/gbm_spatial_demo.yaml
```
> **红线检查**：若 Visium 与 scRNA-seq 参考集的交集基因 `< 8000`（常见于一边为 `EGFR` Symbol、另一边为 `ENSG...` Ensembl ID），脚本会立即报错并尝试自动映射 `.var` 中的 Symbol 列。

### Step 3：`cell2location` 空间细胞解卷积（对齐 Paper 3 参数 `N_cells_per_location=30`, `detection_alpha=200`）
```bash
python scripts/03_run_cell2location_deconv.py --config configs/gbm_spatial_demo.yaml
```
> 若输出端出现椒盐噪点告警（解卷积丰度中位 Moran's I `< 0.15`），建议在 `configs/gbm_spatial_demo.yaml` 中核对 `N_cells_per_location` 或将 `detection_alpha` 降至 `20`。

### Step 4：空间生态位（Niche）、NMF 组织微环境（TM）与扩散伪时间（`DPT` / 可选 `scFates`）投影
```bash
python scripts/04_spatial_niches_and_trajectory.py --config configs/gbm_spatial_demo.yaml
```
输出文件位于 `outputs/figures/` 与 `outputs/tables/`：
- `he_overlay_cell_abundances.png`（解卷积细胞状态丰度半透明叠加至 H&E 切片）
- `he_overlay_spatial_niches.png`（空间生态位 Leiden 分区 + NMF Tissue Microenvironments 叠加图）
- `spatial_pseudotime_trajectory.png`（高肿瘤纯度 Spot 的 `DPT` 连续伪时间直接投射回二维组织坐标）
- `niche_composition.csv` & `pseudotime_summary.csv`

---

## 3. 对接官方 `GBM_space_paper` 与 `spaceTree`
完成上述基础 Demo 后，可进一步对照或运行官方仓库代码：
- 细胞丰度与 NMF Tissue Microenvironments：`../../external/GBM_space_paper/code/2.SPATIAL_TMs/Analysis/2.3.cell2location_NMF.ipynb`
- 发育样 → 胶质增生 → 缺氧时空轨迹：`../../external/GBM_space_paper/code/5.CELLULAR_TRAJECTORIES/`
- （可选）若运行 `setup_repos.sh --with-spacetree`，可用 `../../external/spaceTree` 执行克隆与细胞状态联合空间映射。
