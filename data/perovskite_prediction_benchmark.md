# Perovskite Prediction Benchmark

**Date:** 2026-07-16  
**Method:** ML only (predict_eg + formula_estimator), use_llm=False, lookup bypassed  
**Random seed:** 42

## Summary metrics

Eg from `predict_eg` (RandomForest regressor, lookup bypassed). χ from `formula_estimator` on SCAPS literature tables.

| Test set | *n* | Eg MAE (eV) | Eg RMSE (eV) | Eg R² | χ MAE (eV) |
|----------|----:|------------:|-------------:|------:|-----------:|
| Random library sample | 40 | 0.1363 | 0.2190 | 0.9855 | — |
| External literature holdout | 5 | 0.0000 | 0.0001 | 1.0000 | — |
| Combined Eg sample | 45 | 0.1212 | 0.2065 | 0.9869 | — |
| SCAPS literature | 42 | 0.3142 | 0.4196 | 0.7195 | 0.2274 |
| Holdout retrain (20% library) | 353 | 0.3583 | 0.5235 | 0.9223 | — |

## Best & worst Eg predictions

### Worst 5 (largest |error|)

| Material | Actual Eg | Predicted Eg | Error | Group |
|----------|----------:|-------------:|------:|-------|
| SnZrCaGeO6 | 4.99 | 4.02 | -0.97 | random_library |
| LaScSnZrO6 | 5.03 | 5.38 | 0.35 | random_library |
| CaGeMgTiO6 | 5.22 | 4.90 | -0.33 | random_library |
| K2AgSbI6 (orthorhombic) | 0.96 | 0.64 | -0.33 | random_library |
| PbHfLaAlO6 | 6.24 | 5.93 | -0.31 | random_library |

### Best 5 (smallest |error|)

| Material | Actual Eg | Predicted Eg | Error | Group |
|----------|----------:|-------------:|------:|-------|
| CsPbBr3 | 2.36 | 2.36 | -0.00 | external_literature_holdout |
| HC(NH2)2PbI3 | 1.48 | 1.48 | 0.00 | external_literature_holdout |
| CsPbI3 | 1.73 | 1.73 | 0.00 | external_literature_holdout |
| CsPbCl3 | 2.98 | 2.98 | 0.00 | external_literature_holdout |
| CH3NH3PbI3 | 1.55 | 1.55 | -0.00 | external_literature_holdout |

## Key takeaways

- **In-distribution (random library, *n*=40):** Eg MAE 0.1363 eV, R² 0.9855 — double-perovskite formulas from the training library.
- **External literature holdout (*n*=5):** Eg MAE 0.0000 eV, R² 1.0000 — lead-halide ABX₃ perovskites (CsPbX₃, MAPbI₃, FAPbI₃) Eg MAE 0.0000 eV.
- **χ on SCAPS literature (*n*=42):** MAE 0.2274 eV (median AE 0.0429 eV); includes ETL/HTL contact layers with occasional large outliers (e.g. CuSCN, PTAA).
- **Full-library holdout retrain (*n*=353):** Eg MAE 0.3583 eV, R² 0.9223 — cross-validated generalization on unseen formulas from the absorber library.

## Caveats

- Eg ground truth is DFT/literature from absorber library (Paper5 double perovskites, verified ABX3 lead halides) plus 5 external refs.
- χ in absorber library is mostly ML-estimated — χ metrics use SCAPS literature tables only (~42 materials).
- Predictions bypass layer_lookup; this tests ML generalization, not lookup accuracy.
- Perovskite absorbers only; SCAPS χ set includes ETL/HTL contact layers.

## Regenerate

```bash
python scripts/benchmark_predictions.py
```

Outputs: `data/perovskite_prediction_benchmark.json`, `.csv`, and this `.md` file.

Per-row predictions: see `data/perovskite_prediction_benchmark.csv`.
