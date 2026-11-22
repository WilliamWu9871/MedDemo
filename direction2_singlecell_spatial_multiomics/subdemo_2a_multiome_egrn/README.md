# Sub-demo 2A：配对 RNA+ATAC 联合表征与 eGRN 推断（对应 Paper 2）

对应论文：[Decoding the regulatory programs governing glioblastoma plasticity.pdf](../../Decoding%20the%20regulatory%20programs%20governing%20glioblastoma%20plasticity.pdf)（*Saraswat et al., Nature 2026*）。  
核心复用仓库：`bioFAM/scDoRI`（v2.1+，含 `cisBP` 调控基序库与官方三阶段 Notebook）+ `PMBio/GBM_analysis`（可塑性得分、TAP/TRS 状态跃迁潜能与 `topic_regulation.py`）+ `scvi-tools`（`MultiVI` 纯深度表征基线）。

---

## 1. 快速合成闭环测试（无需下载大数据集）

```bash
cd direction2_singlecell_spatial_multiomics/subdemo_2a_multiome_egrn

python scripts/01_prepare_multiome_mudata.py --source synthetic
python scripts/02_run_multivi_baseline.py --synthetic
python scripts/04_sanity_check_compare.py --synthetic
```

---

## 2. 真实 10x PBMC 3k/10k Multiome 端到端实验（推荐首发调试）

### Step 1：双模态质控、`chr:start-end` 峰名标准化与 `MuData` 组装
```bash
cd direction2_singlecell_spatial_multiomics/subdemo_2a_multiome_egrn
python scripts/01_prepare_multiome_mudata.py --config configs/pbmc_demo.yaml --source pbmc3k
```
> 原始整数计数矩阵保留在 `.X`（供 `MultiVI` 与 `scDoRI` 使用）；`log1p_norm` 与 `tfidf` 存入 `.layers`。

### Step 2：`MultiVI` 纯深度多模态 VAE 基线
```bash
python scripts/02_run_multivi_baseline.py --config configs/pbmc_demo.yaml
```

### Step 3：直接运行官方 `bioFAM/scDoRI` Notebook（v2.0+ 缓存加速模式）
`configs/scdori_pbmc_hg38.json` 已按官方 `scdori.example.json` 对齐人类 `hg38` 与 `"freeze_phase1_in_grn": true, "cache_frozen_grn": true`：
```bash
bash scripts/03_run_scdori.sh --config configs/scdori_pbmc_hg38.json --stage all
```

### Step 4：表征分离度、Topic 塌缩红线检测与 eGRN 自洽性核验
```bash
python scripts/04_sanity_check_compare.py --config configs/pbmc_demo.yaml --scdori-config configs/scdori_pbmc_hg38.json
```
输出文件位于 `outputs/figures/` 与 `outputs/tables/`：
- `umap_multivi_vs_scdori.png`（`MultiVI` vs `scDoRI` 联合 UMAP、Silhouette、ARI）
- `topic_activity_diagnostics.png` & `topic_diagnostics.csv`（激活 Topic 数、香农熵、塌缩红线核查）
- `egrn_consistency_dotplot.png` & `top_topics_egrn.csv`（Top-2 Topic 的核心 TF 与靶基因表达核验 DotPlot）

---

## 3. 切换为 GBM Multiome 子集与对接 `PMBio/GBM_analysis`

若已下载 Paper 2 公开数据（ArrayExpress `E-MTAB-17183` 或 `gbmspace.org`），可截取 3,000–5,000 个细胞运行：
```bash
python scripts/01_prepare_multiome_mudata.py \
  --source h5ad \
  --rna /path/to/gbm_rna.h5ad \
  --atac /path/to/gbm_atac.h5ad \
  --subsample-cells 4000 \
  --cell-type-col cell_state

bash scripts/03_run_scdori.sh --config configs/scdori_gbm_hg38.json --stage all
```
完成 Topic 与 eGRN 推断后，可直接打开或复用官方下游分析仓库进行**表观遗传可塑性评分**与 **TAP/TRS 状态跃迁计算**：
- `../../external/GBM_analysis/plasticity_analysis.ipynb`
- `../../external/GBM_analysis/TAP_TRS_analysis.ipynb`
- `../../external/GBM_analysis/python_scripts/topic_regulation.py`
