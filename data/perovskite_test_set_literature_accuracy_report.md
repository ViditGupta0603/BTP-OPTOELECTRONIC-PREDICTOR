# Literature perovskite test set — OptoStack accuracy

**Date:** 2026-07-25  
**Dataset:** `data/perovskite_test_set_literature.csv`  
**Method:** `predict_stack(..., use_llm=False)` + family/Vegard+ML formula estimator  
**Ground truth:** literature-extracted absorber Eg from `perovskite_test_set_literature.csv` (DOI / paper quotes).  

## Coverage

| | n |
|---|---:|
| Raw CSV rows | 30 |
| Skipped (no absorber / no Eg) | 0 |
| Scored rows (known Eg + predicted) | 30 |
| Unique absorbers (normalized) | 18 |
| In `layer_lookup` | 24 |
| Unseen (not in lookup) | 6 |
| Blocked by perovskite screen | 0 |
| % predicted of scored | 100.0% |
| % scored of raw | 100.0% |

### Skipped rows

None.

## Summary metrics (Eg)

| Evaluation | n | MAE (eV) | RMSE (eV) | R² |
|---|---:|---:|---:|---:|
| Tool vs literature — all scored rows | 30 | 0.1404 | 0.2104 | 0.6777 |
| Tool — in layer_lookup | 24 | 0.1169 | 0.1909 | 0.7667 |
| Tool — unseen (not in lookup) | 6 | 0.2346 | 0.2749 | -1.3638 |
| Tool — unique absorbers | 18 | 0.1686 | 0.2399 | 0.5256 |
| Tool unique — lookup | 13 | 0.1520 | 0.2326 | 0.6426 |
| Tool unique — unseen | 5 | 0.2118 | 0.2577 | -1.2033 |
| ML `predict_eg` — all scored rows | 30 | 0.1743 | 0.2578 | 0.5163 |
| ML `predict_eg` — unseen | 6 | 0.3475 | 0.4170 | -4.4423 |
| ML `predict_eg` — lookup materials | 24 | 0.1310 | 0.1989 | 0.7467 |
| ML — unique absorbers | 18 | 0.2006 | 0.2821 | 0.3439 |

**Hit-rate tool |error| ≤ 0.2 / 0.3 / 0.5 eV (scored rows):** 73.3% / 76.7% / 96.7%

**Hit-rate (unique absorbers) |error| ≤ 0.2 / 0.3 eV:** 66.7% / 72.2%

**Lookup self-consistency** (tool vs stored lookup): MAE = 0.0 eV.  
**Lookup vs literature** (same subset): MAE = 0.1169 eV.

## Type / suitability

- **Type accuracy:** not scored — dataset has no Type-I/II/III ground-truth columns. Predicted types are in the predictions CSV.
- **Suitability accuracy:** not scored — no suitability labels in dataset.

## Model train / eval stats (reference)

| Source | Metric | Value |
|---|---|---:|
| `train_meta.json` | layers in lookup | 1699 |
| `train_meta.json` | Eg holdout MAE (eV) | 0.2926443150057161 |
| `train_meta.json` | Eg CV MAE (eV) | 0.6742113526057629 |
| `train_meta.json` | Eg n | 1898 |
| `train_meta.json` | formula Eg holdout MAE | 0.3309 |
| `train_meta.json` | Type ETL/HTL holdout acc | 1.0 / 1.0 |
| `model_eval_report.json` | Eg holdout MAE / RMSE / R² | 0.33064534677479956 / 0.44585161887416647 / 0.949291027591133 |
| `model_eval_report.json` | Type ETL/HTL holdout acc | 1.0 / 1.0 |

## Worst / best tool Eg predictions (scored rows)

### Worst 5

| Material | Actual | Tool | Error | Source | Lookup? |
|---|---:|---:|---:|---|---|
| CsPbBr3 | 1.793 | 2.360 | +0.567 | lookup | True |
| Cs3Bi2I9 | 2.420 | 2.000 | -0.420 | lookup | True |
| KSnI3 | 1.840 | 1.426 | -0.414 | vegard_plus_ml | False |
| CsSnI3 | 0.950 | 1.300 | +0.350 | lookup | True |
| RbGeI3 | 1.310 | 1.658 | +0.348 | vegard_plus_ml | False |

### Best 5

| Material | Actual | Tool | Error | Source | Lookup? |
|---|---:|---:|---:|---|---|
| CH3NH3PbI3 | 1.550 | 1.550 | +0.000 | lookup | True |
| CH3NH3PbI3 | 1.550 | 1.550 | +0.000 | lookup | True |
| CH3NH3SnI3 | 1.300 | 1.300 | +0.000 | lookup | True |
| CsSnI3 | 1.300 | 1.300 | +0.000 | lookup | True |
| Cs2AgBiBr6 | 2.190 | 2.190 | +0.000 | lookup | True |

## Per-row results

| # | Material | Actual Eg | Tool Eg | |err| | Source | ML Eg | Lookup? | ETL | HTL |
|---:|---|---:|---:|---:|---|---:|---|---|---|
| 0 | CH3NH3PbI3 | 1.550 | 1.550 | 0.000 | lookup | 1.550 | True | TiO2 | Spiro-MeOTAD |
| 1 | CH3NH3PbI3 | 1.550 | 1.550 | 0.000 | lookup | 1.550 | True | TiO2 | Spiro-MeOTAD |
| 2 | CsPbI3 | 1.700 | 1.730 | 0.030 | lookup | 1.730 | True | TiO2 | Spiro-MeOTAD |
| 3 | CsPbI3 | 1.700 | 1.730 | 0.030 | lookup | 1.730 | True | ZnO | NiO |
| 4 | KSnI3 | 1.840 | 1.426 | 0.414 | vegard_plus_ml | 1.304 | False | CeO2 | CBTS |
| 5 | RbGeI3 | 1.310 | 1.658 | 0.348 | vegard_plus_ml | 1.900 | False | TiO2 | Spiro-OMeTAD |
| 6 | RbGeI3 | 1.310 | 1.658 | 0.348 | vegard_plus_ml | 1.900 | False | ZnSe | CuSCN |
| 7 | (FAPbI3)0.85(MAPbBr3)0.15 | 1.550 | 1.629 | 0.079 | vegard_plus_ml | 1.495 | False | ZnO | Cu2O |
| 8 | CsPbI3 | 1.700 | 1.730 | 0.030 | lookup | 1.730 | True | TiO2 | CBTS |
| 9 | CsPbI3 | 1.700 | 1.730 | 0.030 | lookup | 1.730 | True | ZnO | CBTS |
| 10 | CsPbBr3 | 1.793 | 2.360 | 0.567 | lookup | 2.360 | True | TiO2 | Spiro-MeOTAD |
| 11 | Cs3Bi2I9 | 2.420 | 2.000 | 0.420 | lookup | 2.000 | True | TiO2 | Spiro-MeOTAD |
| 12 | CH3NH3SnI3 | 1.300 | 1.300 | 0.000 | lookup | 1.300 | True | TiO2 | Spiro-MeOTAD |
| 13 | HC(NH2)2SnI3 | 1.410 | 1.350 | 0.060 | lookup | 1.350 | True | TiO2 | Spiro-MeOTAD |
| 14 | Cs0.05FA0.85MA0.10PbI3 | 1.460 | 1.500 | 0.040 | vegard_plus_ml | 1.554 | False | SnO2 | Spiro-MeOTAD |
| 15 | CH3NH3PbI3 | 1.560 | 1.550 | 0.010 | lookup | 1.550 | True | TiO2 | Spiro-OMeTAD |
| 16 | CsSnI3 | 1.300 | 1.300 | 0.000 | lookup | 1.300 | True | TiO2 | Spiro-MeOTAD |
| 17 | CsSnI3 | 0.950 | 1.300 | 0.350 | lookup | 1.300 | True | TiO2 | Spiro-MeOTAD |
| 18 | CsSnI3 | 1.130 | 1.300 | 0.170 | lookup | 1.300 | True | PCBM | NiOx |
| 19 | Cs2AgBiI6 | 1.120 | 0.782 | 0.338 | lookup | 0.794 | True | TiO2 | Spiro-MeOTAD |
| 20 | Cs2AgSbCl6 | 1.400 | 1.346 | 0.054 | lookup | 1.615 | True | TiO2 | Spiro-MeOTAD |
| 21 | CsGeI3 | 1.363 | 1.600 | 0.237 | lookup | 1.600 | True | TiO2 | Spiro-MeOTAD |
| 22 | CsSnGeI3 | 1.500 | 1.322 | 0.178 | vegard_plus_ml | 1.280 | False | SnS2 | Cu2O |
| 23 | CsPbBr3 | 2.300 | 2.360 | 0.060 | lookup | 2.360 | True | TiO2 | Spiro-MeOTAD |
| 24 | CH3NH3PbBr3 | 2.200 | 2.320 | 0.120 | lookup | 2.320 | True | TiO2 | Spiro-MeOTAD |
| 25 | Cs3Sb2I9 | 2.010 | 2.050 | 0.040 | lookup | 2.051 | True | TiO2 | Spiro-MeOTAD |
| 26 | Cs3Sb2I9 | 2.010 | 2.050 | 0.040 | lookup | 2.051 | True | ZnO0.25S0.75 | Spiro-MeOTAD |
| 27 | Cs2AgBiBr6 | 2.080 | 2.190 | 0.110 | lookup | 1.981 | True | TiO2 | Spiro-OMeTAD |
| 28 | Cs2AgBiBr6 | 2.080 | 2.190 | 0.110 | lookup | 1.981 | True | TiO2 | Cu2O |
| 29 | Cs2AgBiBr6 | 2.190 | 2.190 | 0.000 | lookup | 1.981 | True | TiO2 | Spiro-OMeTAD |

## Notes

- Default ETL/HTL when unspecified: `TiO2` / `Spiro-MeOTAD` (Eg prediction does not depend on contacts; Type labels may).
- Dual Eg values: HSE preferred over PBE; experimental preferred when labeled; ranges use midpoint. Filled Eg rows cite `eg_fill_doi` / `completion_notes`.
- Dataset completed 2026-07-23: previously skipped rows now have single-absorber formulas + literature Eg (see `gap_method` / `eg_fill_doi` columns).

## Files

- Dataset: `data/perovskite_test_set_literature.csv`
- Predictions: `data/perovskite_test_set_literature_predictions.csv`
- This report: `data/perovskite_test_set_literature_accuracy_report.md`
- JSON: `data/perovskite_test_set_literature_accuracy_report.json`

## Regenerate

```bash
python scripts/eval_literature_test_set.py
```
