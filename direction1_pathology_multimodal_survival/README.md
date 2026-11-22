# 方向一：计算病理与先验引导的多模态生存预测 Demo（对应 Paper 1）

对应论文：[Token-guided multimodal prognosis in hepatocellular carcinom.pdf](../Token-guided%20multimodal%20prognosis%20in%20hepatocellular%20carcinom.pdf)（*Huang et al., Gut 2026*）。  
核心复用仓库：`mahmoodlab/TRIDENT`（WSI 前景分割、20× 切块与 `conch_v15` / `uni_v1` / `resnet50` 离线特征提取）+ `mahmoodlab/SurvPath`（`GatedABMIL`、4 区间离散时间 `NLLSurvLoss` 与多模态 Token Cross-Attention 架构范式）。

---

## 1. 环境准备（独立 Conda 环境或 Google Colab）

* **Google Colab Pro (`L4`) 一键启动**：直接上传并打开 [colab_launcher.ipynb](./colab_launcher.ipynb)（内置挂载 Google Drive、实时落盘日志供本地 Agent Debug、`/content/` 高速盘切块与 `.h5` 特征自动备份）。
* **本地 / Linux 服务器 Conda 环境**：

```bash
conda env create -f direction1_pathology_multimodal_survival/env/environment.yml
conda activate meddemo_patho

# 在 Linux GPU 服务器上按需确保 PyTorch CUDA 版本匹配：
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 拉取并安装 TRIDENT 与 SurvPath（加 --with-patch-encoders 以支持 conch_v15 / ctranspath）
bash direction1_pathology_multimodal_survival/scripts/setup_repos.sh --with-patch-encoders
```

---

## 2. 快速验证模式（`--synthetic` 合成闭环，无需下载数十 GB SVS）

用于快速验证**患者级防泄漏划分、Discrete NLLSurvLoss、ABMIL vs Token-Guided Cross-Attention 消融、RCS 非线性检验与热力图叠加**：

```bash
cd direction1_pathology_multimodal_survival

python scripts/00_download_tcga_subset.py --synthetic --n-patients 72
python scripts/01_create_patient_splits.py --synthetic
python scripts/03_compute_pseudo_prior_tsr.py --synthetic
python scripts/04_train_ablation.py --synthetic --epochs 12
python scripts/05_evaluate_and_explain.py --synthetic
```

输出物位于 `outputs/synthetic/report/`：
- `ablation_cindex_table.csv` / `.md`（含训练集 vs 验证集防泄漏红线核查）
- `km_survival_curves.png`（高/低风险中位数分组 KM 曲线 + Log-rank p 值）
- `multivariable_cox.csv` & `multivariable_cox_forest.png`
- `rcs_tsr_curve.png`（4 节点限制性立方样条验证 TSR 倒 U 型风险曲线）
- `shap_summary.png`
- `heatmaps/*_overlay.png`（Exp 1.1 vs Exp 1.2b 注意力对比与背景占比审计）

---

## 3. 真实 TCGA-LIHC（或 TCGA-GBM）最小实验流程

默认配置为 `configs/tcga_lihc_demo.yaml`（对应 Paper 1 肝癌验证队列）；若跑脑胶质瘤请替换为 `--config configs/tcga_gbm_demo.yaml`。

### Stage 0：拉取 60–80 例 Open Access 诊断级切片（仅 `-DX`，剔除 `-TS` 冷冻切片）
```bash
cd direction1_pathology_multimodal_survival
python scripts/00_download_tcga_subset.py --config configs/tcga_lihc_demo.yaml --n-patients 70 --download
```

### Stage 1：严格患者级 `StratifiedGroupKFold`（5 折 × 3 种子）
```bash
python scripts/01_create_patient_splits.py --config configs/tcga_lihc_demo.yaml
```

### Stage 2：TRIDENT 一键组织分割、切块与离线特征提取
```bash
# 若 HuggingFace 尚未批准 conch_v15/uni_v1 门控权重，可先改用 --patch-encoder resnet50 或 ctranspath
bash scripts/02_extract_trident_features.sh \
  --wsi-dir data/tcga_lihc/svs \
  --job-dir outputs/tcga_lihc/trident_feats \
  --patch-encoder conch_v15 \
  --mag 20 \
  --patch-size 256
```

> **提示**：TRIDENT 会将 Patch 特征保存至 `outputs/tcga_lihc/trident_feats/20x_256px/features_<encoder>/`。运行 Stage 3B 时请用 `--feats-dir` 指向该特征目录（或对应路径）。

### Stage 3A & 3B：伪先验 TSR Token 构造 + 严格对齐的 5 折消融训练
```bash
python scripts/03_compute_pseudo_prior_tsr.py --config configs/tcga_lihc_demo.yaml

python scripts/04_train_ablation.py \
  --config configs/tcga_lihc_demo.yaml \
  --feats-dir outputs/tcga_lihc/trident_feats/20x_256px/features_conch_v15 \
  --all-seeds
```

### Stage 4：生存统计检验与可解释性热力图验收
```bash
python scripts/05_evaluate_and_explain.py --config configs/tcga_lihc_demo.yaml
```
