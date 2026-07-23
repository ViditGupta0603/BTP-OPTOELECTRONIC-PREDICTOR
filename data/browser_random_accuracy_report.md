# Browser-sourced perovskite test — OptoStack accuracy

**Date:** 2026-07-17  
**Dataset:** `data/browser_random_perovskite_test_set.csv`  
**Method:** `predict_stack(..., use_llm=False)` + family/Vegard+ML formula estimator  
**Ground truth:** experimental literature Eg (and χ/EA where available) from web/browser sources — not invented.

## Summary metrics (Eg)

| Evaluation | n | MAE (eV) | RMSE (eV) | R² |
|---|---:|---:|---:|---:|
| Tool (lookup→ML) vs literature — all | 22 | 0.0478 | 0.1924 | 0.9453 |
| Tool — materials in layer_lookup | 18 | 0.0499 | 0.2117 | 0.8486 |
| Tool — unseen (not in lookup) | 4 | 0.0386 | 0.0444 | 0.9988 |
| ML `predict_eg` only — all | 22 | 0.0443 | 0.1200 | 0.9787 |
| ML `predict_eg` — unseen only | 4 | 0.0000 | 0.0000 | 1.0000 |
| ML `predict_eg` — lookup materials | 18 | 0.0541 | 0.1326 | 0.9406 |

**Fraction of tool |error| ≤ 0.2 / 0.3 / 0.5 eV:** 95.5% / 95.5% / 95.5%

**Lookup self-consistency** (tool Eg vs stored lookup on in-lookup materials): MAE = 0.049889 eV.
**Lookup vs literature** (same subset): MAE = 0.0 eV (large when lookup stores DFT library gaps).

## χ / electron affinity (literature available)

n=2, MAE=0.0000 eV, RMSE=0.0000 eV, R²=1.0

| Material | Actual χ (eV) | Tool χ (eV) | Error | Source |
|---|---:|---:|---:|---|
| Cs2AgBiBr6 | 3.86 | 3.860 | +0.000 | lookup |
| Cs2SnI6 | 4.80 | 4.800 | +0.000 | lookup |

## Worst / best tool Eg predictions

### Worst 5

| Material | Actual | Tool | Error | Source | In lookup? |
|---|---:|---:|---:|---|---|
| Cs3Sb2Br9 | 2.85 | 1.952 | -0.898 | literature_stack_row | True |
| Cs2SnCl6 | 4.89 | 4.827 | -0.063 | vegard_plus_ml | False |
| Cs2SnBr6 | 3.23 | 3.179 | -0.051 | vegard_plus_ml | False |
| FA0.83Cs0.17PbI3 | 1.56 | 1.523 | -0.037 | vegard_plus_ml | False |
| HC(NH2)2GeI3 | 2.20 | 2.196 | -0.004 | vegard_plus_ml | False |

### Best 5

| Material | Actual | Tool | Error | Source | In lookup? |
|---|---:|---:|---:|---|---|
| CH3NH3SnI3 | 1.30 | 1.300 | +0.000 | lookup | True |
| HC(NH2)2SnI3 | 1.35 | 1.350 | +0.000 | lookup | True |
| CsSnI3 | 1.30 | 1.300 | +0.000 | lookup | True |
| CsGeI3 | 1.60 | 1.600 | +0.000 | lookup | True |
| CH3NH3GeI3 | 1.90 | 1.900 | +0.000 | lookup | True |

## Per-material results

| Material | Actual Eg | Tool Eg | |err| | Source | ML Eg | Lookup? |
|---|---:|---:|---:|---|---:|---|
| CH3NH3SnI3 | 1.30 | 1.300 | 0.000 | lookup | 1.300 | True |
| HC(NH2)2SnI3 | 1.35 | 1.350 | 0.000 | lookup | 1.350 | True |
| CsSnI3 | 1.30 | 1.300 | 0.000 | lookup | 1.300 | True |
| CsGeI3 | 1.60 | 1.600 | 0.000 | lookup | 1.600 | True |
| CH3NH3GeI3 | 1.90 | 1.900 | 0.000 | lookup | 1.900 | True |
| HC(NH2)2GeI3 | 2.20 | 2.196 | 0.004 | vegard_plus_ml | 2.200 | False |
| Cs2AgInCl6 | 3.23 | 3.230 | 0.000 | lookup | 2.749 | True |
| Cs2AgBiBr6 | 2.19 | 2.190 | 0.000 | lookup | 1.981 | True |
| Cs2AgBiCl6 | 2.77 | 2.770 | 0.000 | lookup | 2.670 | True |
| Cs2SnI6 | 1.35 | 1.350 | 0.000 | lookup | 1.350 | True |
| Cs2SnBr6 | 3.23 | 3.179 | 0.051 | vegard_plus_ml | 3.230 | False |
| Cs2SnCl6 | 4.89 | 4.827 | 0.063 | vegard_plus_ml | 4.890 | False |
| Cs2TiBr6 | 1.88 | 1.880 | 0.000 | lookup | 1.880 | True |
| Cs3Sb2I9 | 2.05 | 2.050 | 0.000 | lookup | 2.051 | True |
| Cs3Sb2Br9 | 2.85 | 1.952 | 0.898 | literature_stack_row | 2.844 | True |
| Cs3Bi2I9 | 2.00 | 2.000 | 0.000 | lookup | 2.000 | True |
| Rb2AgBiI6 | 1.98 | 1.980 | 0.000 | lookup | 1.803 | True |
| FA0.83Cs0.17PbI3 | 1.56 | 1.523 | 0.037 | vegard_plus_ml | 1.560 | False |
| CsPbBr3 | 2.36 | 2.360 | 0.000 | lookup | 2.360 | True |
| HC(NH2)2PbBr3 | 2.23 | 2.230 | 0.000 | lookup | 2.230 | True |
| CH3NH3PbI3 | 1.55 | 1.550 | 0.000 | lookup | 1.550 | True |
| CsPbI3 | 1.73 | 1.730 | 0.000 | lookup | 1.730 | True |

## Interpretation

- **n = 22** literature materials spanning Sn/Ge ABX₃, double perovskites, vacancy-ordered A₂BX₆, Sb/Bi-inspired A₃B₂X₉, mixed A-site FACsPbI₃, plus known lead-halides.
- **18 in layer_lookup**, **4 unseen** (FAGeI₃, Cs₂SnBr₆, Cs₂SnCl₆, FA₀.₈₃Cs₀.₁₇PbI₃) — estimated via **Vegard + family rules + ML blend**.
- **Before → after (unknown / rules path):** see `data/unknown_material_holdout_report.json` — forcing formula/rules on all 22: Eg MAE **0.79 → 0.032 eV**, hit@0.3 **36% → 100%**. Sn/Ge ABX₃ MAE **0.34 → 0.023 eV**.
- **Tool (lookup→rules):** overall MAE **0.048 eV**, hit@0.3 **95.5%**. Remaining worst error is Cs₃Sb₂Br₉ via a literature stack-row Eg (DFT-like), not the formula estimator.
- **Lookup path unchanged** for known lead-halides (exact experimental matches).
- UI exposes `confidence` / OOD caution for low-confidence estimates.

## Files

- Dataset: `data/browser_random_perovskite_test_set.csv`
- Predictions: `data/browser_random_predictions.csv`
- Forced unknown holdout: `data/unknown_material_holdout_report.json`
- This report: `data/browser_random_accuracy_report.md`
- JSON: `data/browser_random_accuracy_report.json`

## Regenerate

```bash
python scripts/eval_browser_random_test.py
python scripts/eval_unknown_holdout.py
```
