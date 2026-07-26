# Round 3 fix verification

Branch: `fix/round3-a2bx6-halide-types`

## P1 — A2BX6 halide identity (Cs2TiX6)

- **Before (Round 2 / eval):** Cs2TiI6 Eg=**1.800** eV (collided with Br≈1.8)
- **After:** Cs2TiI6 Eg=**1.58** eV (expect 1.56–1.65): **PASS**
- **After:** Cs2TiBr6 Eg=**1.88** eV (expect ≈1.88, distinct): **PASS**
- **After:** Cs2TiCl6 Eg=**2.9** eV

## P1 — Exact-match resolve (no prefix)

- `Cs2TiI6` → Eg=1.58 (norm=`Cs2TiI6`): **PASS**
- `Cs2TiBr6` → Eg=1.88 (norm=`Cs2TiBr6`): **PASS**
- `Cs2TiCl6` → Eg=2.9 (norm=`Cs2TiCl6`): **PASS**
- Cs2TiI6 ≠ Cs2TiBr6 (|ΔEg|=0.300): **PASS**

## P1 — Halide monotonicity Eg(I) < Eg(Br) < Eg(Cl)

| Series | Eg(I) | Eg(Br) | Eg(Cl) | Verdict |
|---|---:|---:|---:|---|
| Cs2TiI6 / Cs2TiBr6 / Cs2TiCl6 | 1.58 | 1.88 | 2.9 | **PASS** |
| K2TiI6 / K2TiBr6 / K2TiCl6 | 1.61 | 1.85 | 2.85 | **PASS** |
| Rb2TiI6 / Rb2TiBr6 / Rb2TiCl6 | 1.6 | 1.86 | 2.88 | **PASS** |
| Cs2SnI6 / Cs2SnBr6 / Cs2SnCl6 | 1.35 | 3.23 | 4.89 | **PASS** |
| Cs2PdI6 / Cs2PdBr6 / Cs2PdCl6 | 1.2 | 1.67 | 2.4 | **PASS** |
| Cs2PtI6 / Cs2PtBr6 / Cs2PtCl6 | 1.4 | 1.95 | 2.7 | **PASS** |

## P2 — Type I and Type III coverage (published CBM/VBM)

Anderson Types from vacuum edges (`CBM=-χ`, `VBM=-(χ+Eg)`). Typical PSC stacks are Type I/II; Type III (broken gap) appears when a deep-χ absorber meets a shallow-IP contact.

| Stack | Eg | χ | ETL Type | HTL Type | Notes |
|---|---:|---:|---|---|---|
| Cs2SnI6/TiO2/P3HT | 1.35 | 4.8 | Type I | Type II | ETL expect Type I: OK |
| Cs2SnI6/TiO2/CFTS | 1.35 | 4.8 | Type I | Type III | HTL expect Type III: OK |
| MAPbI3/TiO2/Spiro-OMeTAD | 1.55 | 3.7946 | Type II | Type II | ETL expect Type II: OK; HTL expect Type II: OK |
| K2TiI6/WS2/NiO | 1.61 | 4.0 | Type I | Type II | ETL expect Type I: OK |
| MAPbI3/TiO2/NiO | 1.55 | 3.7946 | Type II | Type I | HTL expect Type I: OK |

### Direct `junction_type` (CBM/VBM)

- Cs2SnI6/TiO2: `Type I` expect Type I — **PASS**
- Cs2SnI6/CFTS: `Type III` expect Type III — **PASS**
- Synthetic broken-gap: `Type III` expect Type III

Types observed in stacks: `['Type I', 'Type II', 'Type III']` — **PASS** (need I, II, and III).

## P3 — FAPbCl3 toward ~2.9 eV

- FAPbCl3 Eg=2.9 / HC(NH2)2PbCl3 Eg=2.9 (expect ≈2.90): **PASS**

## Summary

```json
{
  "cs2tii6_eg": 1.58,
  "cs2tibr6_eg": 1.88,
  "cs2ticl6_eg": 2.9,
  "cs2tii6_in_range": true,
  "cs2tibr6_distinct": true,
  "exact_resolve_ok": true,
  "monotonicity_ok": true,
  "monotonicity": [
    {
      "series": "Cs2TiI6 / Cs2TiBr6 / Cs2TiCl6",
      "Eg_I": 1.58,
      "Eg_Br": 1.88,
      "Eg_Cl": 2.9,
      "ok": true
    },
    {
      "series": "K2TiI6 / K2TiBr6 / K2TiCl6",
      "Eg_I": 1.61,
      "Eg_Br": 1.85,
      "Eg_Cl": 2.85,
      "ok": true
    },
    {
      "series": "Rb2TiI6 / Rb2TiBr6 / Rb2TiCl6",
      "Eg_I": 1.6,
      "Eg_Br": 1.86,
      "Eg_Cl": 2.88,
      "ok": true
    },
    {
      "series": "Cs2SnI6 / Cs2SnBr6 / Cs2SnCl6",
      "Eg_I": 1.35,
      "Eg_Br": 3.23,
      "Eg_Cl": 4.89,
      "ok": true
    },
    {
      "series": "Cs2PdI6 / Cs2PdBr6 / Cs2PdCl6",
      "Eg_I": 1.2,
      "Eg_Br": 1.67,
      "Eg_Cl": 2.4,
      "ok": true
    },
    {
      "series": "Cs2PtI6 / Cs2PtBr6 / Cs2PtCl6",
      "Eg_I": 1.4,
      "Eg_Br": 1.95,
      "Eg_Cl": 2.7,
      "ok": true
    }
  ],
  "types_seen": [
    "Type I",
    "Type II",
    "Type III"
  ],
  "types_i_ii_iii_ok": true,
  "fapbcl3_ok": true,
  "fapbcl3_eg": 2.9
}
```
