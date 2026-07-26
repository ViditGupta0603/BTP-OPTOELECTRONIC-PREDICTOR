# Round 2 fix verification

Branch: `fix/round2-type-halide-pedot`

## P0 — Halide-sensitive Eg lookup

- `FAPbI3` → `HC(NH2)2PbI3` Eg=1.48 (expect 1.45–1.55): **PASS**
- `FAPbBr3` → `HC(NH2)2PbBr3` Eg=2.23 (expect 2.15–2.35): **PASS**
- `FAPbBr\u2083` → `HC(NH2)2PbBr3` Eg=2.23 (expect 2.15–2.35): **PASS**
- `MAPbI3` → `CH3NH3PbI3` Eg=1.55 (expect 1.5–1.6): **PASS**
- `MAPbBr3` → `CH3NH3PbBr3` Eg=2.32 (expect 2.2–2.4): **PASS**
- `MAPbBr\u2083` → `CH3NH3PbBr3` Eg=2.32 (expect 2.2–2.4): **PASS**

## P0 — Junction Types vary (not all Type I)

| Stack | Eg | ETL Type | HTL Type | Verdict | Notes |
|---|---:|---|---|---|---|
| MAPbI3/TiO2/Spiro-OMeTAD | 1.55 | Type II | Type II | YES | — |
| Cs2SnI6/TiO2/P3HT | 1.35 | Type I | Type II | YES | — |
| K2TiI6/PC60BM/MoO3 | 1.61 | Type II | Type II | YES | — |
| K2TiI6/WS2/NiO | 1.61 | Type I | Type II | YES | — |
| CsSnI3/TiO2/PEDOT:PSS | 1.3 | Type II | Type II | YES | PEDOT caveat; gap_type: degenerate/metallic HTL — PEDOT:PSS is a highly doped polymer; Eg-based Anderson Type is unreliable for this contact. |
| CsPbBr3/ZnO/NiO | 2.36 | Type II | Type II | YES | — |
| FAPbBr3/TiO2/Spiro-OMeTAD | 2.23 | Type II | Type II | YES | — |

Types observed: `['Type I', 'Type II']` — **PASS** (must not be only Type I).

## Anderson sanity (direct `junction_type`)

- narrow-in-wide (Type I): got `Type I` expect `Type I` — **PASS**
- MAPbI3/TiO2 staggered (Type II): got `Type II` expect `Type II` — **PASS**
- broken gap (Type III): got `Type III` expect `Type III` — **PASS**

## P1 — PEDOT:PSS degenerate HTL caveat

- CsSnI3/TiO2/PEDOT:PSS caveat present: **PASS** (verdict=YES, chi_CsSnI3=3.62)

## χ sample (absorber vs contacts)

- `MAPbI3`: {'Eg_eV': 1.55, 'chi_eV': 3.7946}
- `FAPbBr3`: {'Eg_eV': 2.23, 'chi_eV': 3.7725}
- `CsSnI3`: {'Eg_eV': 1.3, 'chi_eV': 3.62}
- `TiO2`: {'Eg_eV': 3.2, 'chi_eV': 4.0}
- `Spiro-OMeTAD`: {'Eg_eV': 3.0, 'chi_eV': 2.05}
- `PEDOT:PSS`: {'Eg_eV': 1.6, 'chi_eV': 3.3}
- `NiO`: {'Eg_eV': 3.6, 'chi_eV': 1.8}
- `ZnO`: {'Eg_eV': 3.3, 'chi_eV': 4.0}

## Summary

```json
{
  "halide_ok": true,
  "types_vary": true,
  "types_seen": [
    "Type I",
    "Type II"
  ],
  "pedot_caveat": true
}
```
