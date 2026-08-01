# Anderson Type I / II / III verification

**Date:** 2026-08-01  
**Convention:** `CBM = -χ`, `VBM = -(χ + Eg)` (eV vs vacuum)  
**Type III test:** `VBM_a ≥ CBM_b or VBM_b ≥ CBM_a` (gaps do not overlap)

## Direct `junction_type` on hand-computed edges

| Case | CBM/VBM A | CBM/VBM B | Got | Expect | Result |
|---|---|---|---|---|---|
| InAs/GaSb — textbook broken gap (GaSb VBM above InAs CBM) | -4.90/-5.25 | -4.06/-4.79 | `Type III` | `Type III` | **PASS** |
| MAPbI3/MoO3 — deep-affinity oxide CBM below absorber VBM | -3.90/-5.45 | -6.70/-9.70 | `Type III` | `Type III` | **PASS** |
| CsPbI3/V2O5 — deep-affinity oxide | -3.95/-5.68 | -6.60/-9.40 | `Type III` | `Type III` | **PASS** |
| GaAs/AlGaAs — straddling | -4.07/-5.49 | -3.74/-5.54 | `Type I` | `Type I` | **PASS** |
| MAPbI3/MgO — wide-gap insulator straddles absorber gap | -3.90/-5.45 | -0.85/-8.65 | `Type I` | `Type I` | **PASS** |
| MAPbI3/TiO2 — staggered | -3.90/-5.45 | -4.00/-7.20 | `Type II` | `Type II` | **PASS** |
| Zero-overlap boundary (absorber VBM exactly at partner CBM) | -4.00/-5.00 | -5.00/-6.00 | `Type III` | `Type III` | **PASS** |

## Contact band edges available to the pipeline

| Material | Eg (eV) | χ (eV) | Expected | Result |
|---|---:|---:|---|---|
| MoO3 | 3.0 | 6.7 | Eg=3.0 χ=6.7 | **PASS** |
| V2O5 | 2.8 | 6.6 | Eg=2.8 χ=6.6 | **PASS** |
| WO3 | 3.1 | 5.0 | Eg=3.1 χ=5.0 | **PASS** |
| MgO | 7.8 | 0.85 | Eg=7.8 χ=0.85 | **PASS** |
| Al2O3 | 8.8 | 1.35 | Eg=8.8 χ=1.35 | **PASS** |

## End-to-end `predict_stack`

| Stack | ETL Type | HTL Type | Verdict | Result |
|---|---|---|---|---|
| MAPbI3/TiO2/MoO3 | Type II | Type III | MARGINAL | **PASS** |
| CsPbI3/SnO2/V2O5 | Type II | Type III | MARGINAL | **PASS** |
| Cs2AgBiBr6/ZnO/MoO3 | Type II | Type III | MARGINAL | **PASS** |
| FaSnI3/MoO3/MgO | Type III | Type I | MARGINAL | **PASS** |
| FASnI3/MoO3/MgO | Type III | Type I | MARGINAL | **PASS** |
| MAPbI3/TiO2/Spiro-OMeTAD | Type II | Type II | YES | **PASS** |
| K2TiI6/WS2/NiO | Type I | Type II | YES | **PASS** |
| CsSnI3/TiO2/NiO | Type II | Type I | YES | **PASS** |
| FAPbI3/TiO2/CuSCN | Type II | Type I | YES | **PASS** |

## Summary

```json
{
  "types_seen": [
    "Type I",
    "Type II",
    "Type III"
  ],
  "all_three_types_ok": true,
  "chi_clamp": [
    0.3,
    7.2
  ],
  "chi_clamp_admits_deep_affinity": true,
  "n_failures": 0
}
```
