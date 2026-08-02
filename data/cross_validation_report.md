# OptoStack cross-validation report

**Date:** 2026-08-02

**Stack table size:** 1030 rows (`perovskite_stack_dataset.csv`; train_meta type n_etl=1126).
**Absorber library:** 1764 rows (Eg training expands verified_external ×10 → 1908 samples).

## Models in use

| Role | sklearn class | Artifact |
|------|---------------|----------|
| Absorber Eg | `RandomForestRegressor` | `data/models/perovskite_eg_regressor.joblib` |
| Formula Eg | `RandomForestRegressor` | `data/models/formula_eg_chi_estimator.joblib` |
| Formula χ | `GradientBoostingRegressor` | same joblib |
| Type ETL | `GradientBoostingClassifier` (Pipeline) | `data/models/stack_type_classifier.joblib` |
| Type HTL | `GradientBoostingClassifier` (Pipeline) | same joblib |

One-line summary: **Eg RF (500 trees) + formula RF/GBR + Type ETL/HTL GBC pipelines** (all `random_state=42`).

## Hyperparameters

| Model | Parameters |
|-------|------------|
| Absorber Eg RF | n_estimators=500, max_depth=None, min_samples_leaf=1, random_state=42, n_jobs=-1 |
| Formula Eg RF | n_estimators=400, max_depth=20, min_samples_leaf=2, random_state=42 |
| Formula χ GBR | n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42 |
| Type GBC | n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42; features: OneHot(absorber,partner)+scaled Eg diffs / organic flags |

## CV protocol

- **Eg (absorber + formula):** `GroupKFold` (k=5) grouped by material `base_name` — avoids leakage from verified-row oversampling and duplicate names.
- **Type ETL/HTL (primary):** `GroupKFold` by absorber name (realistic for new absorbers).
- **Type (secondary):** `StratifiedKFold` (k=5, shuffle, seed=42) — typically optimistic because categorical material names can be memorized.
- Metrics: Eg → MAE, RMSE, R² per fold + mean±std; Type → accuracy, macro F1, macro OvR ROC-AUC.

## Eg absorber regressor metrics

| Metric | mean ± std (GroupKFold) |
|--------|-------------------------|
| MAE (eV) | 0.3021 ± 0.0066 |
| RMSE (eV) | 0.4762 ± 0.0095 |
| R² | 0.9327 ± 0.0063 |
| Pooled OOF R² | 0.9331 |

### Per-fold Eg

| Fold | n_test | MAE | RMSE | R² |
|------|--------|-----|------|----|
| 1 | 382 | 0.2976 | 0.4696 | 0.9385 |
| 2 | 382 | 0.3030 | 0.4832 | 0.9332 |
| 3 | 382 | 0.3114 | 0.4913 | 0.9230 |
| 4 | 381 | 0.2922 | 0.4708 | 0.9289 |
| 5 | 381 | 0.3061 | 0.4661 | 0.9400 |

## Formula Eg / χ metrics

| Target | MAE | RMSE | R² |
|--------|-----|------|----|
| Formula Eg | 0.3547 ± 0.0191 | 0.5320 ± 0.0277 | 0.9167 ± 0.0117 |
| Formula χ | 0.3111 ± 0.0894 | 0.3844 ± 0.1383 | 0.3719 ± 0.8113 |

## Type classification metrics

### GroupKFold by absorber (preferred)

| Side | Accuracy | macro F1 | macro OvR ROC-AUC |
|------|----------|----------|-------------------|
| ETL | 0.6697 ± 0.2462 | 0.6054 ± 0.2991 | 0.8364 ± 0.1931 |
| HTL | 0.8330 ± 0.0955 | 0.8191 ± 0.1097 | 0.8940 ± 0.0864 |

### StratifiedKFold (comparison — often optimistic)

| Side | Accuracy | macro F1 | macro OvR ROC-AUC |
|------|----------|----------|-------------------|
| ETL | 0.9290 ± 0.0115 | 0.7730 ± 0.0613 | 0.9767 ± 0.0052 |
| HTL | 0.9290 ± 0.0151 | 0.9097 ± 0.0181 | 0.9788 ± 0.0046 |

## Figures

| Figure | Path |
|--------|------|
| Eg predicted vs actual (R²) | `data/figures/eg_r2_scatter.png` |
| Eg fold-wise R² bars | `data/figures/eg_r2_fold_bars.png` |
| Eg residuals | `data/figures/eg_residuals.png` |
| Formula Eg scatter | `data/figures/formula_eg_r2_scatter.png` |
| Type ETL ROC (OvR) | `data/figures/type_etl_roc_ovr.png` |
| Type HTL ROC (OvR) | `data/figures/type_htl_roc_ovr.png` |
| Type ETL confusion heatmap | `data/figures/type_etl_confusion_heatmap.png` |
| Type HTL confusion heatmap | `data/figures/type_htl_confusion_heatmap.png` |

## Caveats

- **Lookup vs ML:** Runtime prefers literature / stack-table lookup and physics Type from Eg+χ. ML is used when library values are missing; these CV scores describe the ML fallbacks, not the lookup path.
- **GroupKFold vs random:** Stratified/random Type accuracy can approach ~1.0 via name memorization; GroupKFold-by-absorber is the realistic number for novel absorbers.
- **Suitability is not CV'd:** YES/MARGINAL/NO comes from deterministic Anderson rules on Types + offsets, not a supervised model.
- **Formula estimator** blends family/Vegard priors with ML at inference; CV here evaluates the ML regressor heads only.
- Holdout MAE in `train_meta.json` uses a single random split and may differ from GroupKFold means.

## Artifact status at run time

```json
{
  "formula": "present",
  "eg": "present",
  "type": "present"
}
```
